# Dual-Draft Remote Speculative Decoding 设计文档

> **版本**: v0.1  
> **状态**: Draft  
> **目标**: 将 1 Target : 1 Draft 升级为 1 Target : 2 Drafts，通过预生成备选路径消除拒绝时的额外 RTT

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [算法思想](#2-算法思想)
3. [详细示例](#3-详细示例)
4. [角色动态交换机制](#4-角色动态交换机制)
5. [数据结构设计](#5-数据结构设计)
6. [通信协议设计](#6-通信协议设计)
7. [各模块实现方案](#7-各模块实现方案)
8. [边界情况与容错](#8-边界情况与容错)
9. [性能分析](#9-性能分析)

---

## 1. 背景与动机

### 1.1 现有架构的瓶颈

当前 1:1 架构的时间线如下：

```
Round N:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Target: [send req to D1] → [GPU verify [d0,d1,d2]] → [output]   │
  │ D1:                         [generate [d3,d4,d5]]                │
  └──────────────────────────────────────────────────────────────────┘

Round N+1 (全部接受，pipeline 命中):
  ┌──────────────────────────────────────────────────────────────────┐
  │ Target: [send req to D1] → [GPU verify [d3,d4,d5]] → [output]   │
  │ D1:                         [generate [d6,d7,d8]]                │
  └──────────────────────────────────────────────────────────────────┘

Round N+1 (拒绝，pipeline 未命中):
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ Target: [GPU verify [d0,d1,d2]] → 拒绝在位置 i → [send retry to D1]       │
  │                                                         ↓                  │
  │                                              [等待 D1 重新生成] ← 额外 RTT │
  │ D1:     [generate [d3,d4,d5]] → 废弃 → [重新 generate from bonus]          │
  └────────────────────────────────────────────────────────────────────────────┘
```

**关键问题**：拒绝时 Target 必须等待 D1 重新生成，引入一个完整的 Draft 推理 RTT（通常 5~20ms）。  
**Accept rate** 越低，这个开销越显著。

### 1.2 核心洞察

Draft 模型在每个 decode step 计算 logit 向量时，获取 top-1 token（主路 token）的同时，**零额外 forward 代价**可以获取 top-k token（备选 token）。

若 Target 拒绝位置 `i` 并产生 correction token `b`，如果 `b` 恰好是 Draft 在该位置的 top-2 或 top-3 candidate，那么从 `b` 出发的后续推理**可以提前由第二个 Draft 服务器完成**，从而消除额外 RTT。

---

## 2. 算法思想

### 2.1 整体架构

```
          ┌──────────────────────────────────────────────┐
          │                  Target Server                │
          │  verify → 判断接受/拒绝 → 仲裁下一轮主路      │
          └──────────┬──────────────────┬────────────────┘
                     │                  │
          ┌──────────▼──────┐  ┌────────▼──────────┐
          │   Draft A (物理) │  │  Draft B (物理)    │
          │  逻辑角色动态分配 │  │  逻辑角色动态分配  │
          └─────────────────┘  └───────────────────┘
```

每个时刻，两个物理 Draft 服务器各自承担一个**逻辑角色**：

| 逻辑角色 | 职责 |
|---------|------|
| **Primary Draft (主路)** | 持有当前被 Target 认可的 KV cache，乐观地沿主路继续生成下 N 个 token |
| **Backup Draft (备路)** | 接收主路的草稿 token + 各位置备选 token，生成覆盖所有拒绝场景的续生序列 |

### 2.2 备选 Token 的生成（零额外 Forward）

Primary Draft 生成主路草稿 `[d0, d1, d2]` 时，**同时**在每个位置记录 top-k 备选：

```
位置 0: primary=d0, alternatives=[x0, y0]   ← 来自 logit[:, pos0] 的 top-3
位置 1: primary=d1, alternatives=[x1, y1]   ← 来自 logit[:, pos1] 的 top-3  
位置 2: primary=d2, alternatives=[x2, y2]   ← 来自 logit[:, pos2] 的 top-3
```

这些 alternatives 随草稿 token 一起发送给 Target 和 Backup Draft，**不需要额外的模型 forward**。

### 2.3 Backup Draft 的树形推理

Backup Draft 收到 `{主路草稿, 各位置 alternatives}` 后，构建以下覆盖树：

```
拒绝位置 0: context + [x0]      → 续生后续 tokens
            context + [y0]      → 续生后续 tokens

拒绝位置 1: context + [d0, x1]  → 续生后续 tokens
            context + [d0, y1]  → 续生后续 tokens

拒绝位置 2: context + [d0,d1,x2]→ 续生后续 tokens
            context + [d0,d1,y2]→ 续生后续 tokens
```

Backup Draft 通过一次**树形 prefill/extend**（类似 EAGLE 的 tree attention）并行生成所有分支的续生 token，整体计算量仍为一次 forward pass。

### 2.4 Target 的决策逻辑

```
verify 结果:
  Case A: 全部接受 [d0,d1,d2]
      → 主路 token 已接受，Primary 的后续草稿 [d3,d4,d5] 直接用于下轮
      → 逻辑角色不变，Primary 继续乐观生成

  Case B: 在位置 i 拒绝，correction token = b
      → 查找: b ∈ alternatives[i]?
        命中: 使用 Backup Draft 对应分支的续生序列，角色交换
        未命中: 走现有 retry 流程（与 1:1 架构相同）
```

---

## 3. 详细示例

以 `num_draft_tokens=3`、`num_alternatives=2`（即每位置取 top-3，primary + 2 alternatives）为例。

### 3.1 初始状态（Round 0）

```
Target output_ids: [p0, p1, p2, ..., pN]   (prompt + 已生成)
Primary = Draft A,  Backup = Draft B
```

---

### 3.2 Round 1：首轮正常 pipeline

**t1 (Draft A 完成生成，触发本轮)**：

```
Draft A 生成草稿:
  [d0, d1, d2]
  并记录备选:
  位置 0: [x0, y0]
  位置 1: [x1, y1]
  位置 2: [x2, y2]

Draft A → Target: {主路=[d0,d1,d2], alternatives={0:[x0,y0], 1:[x1,y1], 2:[x2,y2]}}
Draft A → Draft B: 同上内容，通知 B 开始准备备路
```

**t2 (三方并行)**：

```
Target:   verify([d0, d1, d2])
            ↕ GPU forward (TARGET_VERIFY)

Draft A:  乐观生成 [d3, d4, d5]
            (假设 [d0,d1,d2] 全部接受)

Draft B:  树形推理，生成 6 条备路的续生 token:
            branch(x0): context → [x0] → [b0_x0, b1_x0, b2_x0]
            branch(y0): context → [y0] → [b0_y0, b1_y0, b2_y0]
            branch(x1): context+[d0] → [x1] → [b0_x1, b1_x1, b2_x1]
            branch(y1): context+[d0] → [y1] → [b0_y1, b1_y1, b2_y1]
            branch(x2): context+[d0,d1] → [x2] → [b0_x2, b1_x2, b2_x2]
            branch(y2): context+[d0,d1] → [y2] → [b0_y2, b1_y2, b2_y2]
```

**t3 (Target verify 结果)**：

**场景 A: 全部接受 [d0, d1, d2]，bonus = d3_target**

```
Target 输出: [d0, d1, d2, d3_target]
              ↑ 全部接受     ↑ bonus token

检查 draft A 的 [d3, d4, d5]:
  若 d3 == d3_target → pipeline 命中！
  下轮 Primary=A (持有 [d3,d4,d5] 的 KV), Backup=B
  B 需要为 [d3,d4,d5] 生成新备路

B 在 t2 阶段生成的 6 条备路废弃 (可接受的开销)
```

**场景 B: 在位置 1 拒绝，correction token = x1**

```
Target 输出: [d0, x1, ...]
              ↑接受  ↑correction

查找: x1 ∈ alternatives[1] = [x1, y1] → 命中！

使用 Draft B 的 branch(x1) 续生序列: [b0_x1, b1_x1, b2_x1]
此时:
  - Draft B (branch x1) 的路径被采纳 → B 升级为新 Primary
  - Draft A (原主路废弃) → 降级为新 Backup

角色交换:
  Primary = Draft B  (KV cache 已在 branch(x1) 对应的 extend 中建立)
  Backup  = Draft A  (接收新主路 [x1, b0_x1, b1_x1, b2_x1] + B 生成的新 alternatives)
  
下轮 Target verify: [b0_x1, b1_x1, b2_x1]  (免去额外 RTT！)
```

**场景 C: 在位置 1 拒绝，correction token = z（不在 alternatives 中）**

```
Target 输出: [d0, z, ...]
查找: z ∉ alternatives[1] → 未命中

fallback to retry:
  Target → Primary (Draft A): retry 请求，从 [d0, z] 出发生成新草稿
  (与现有 1:1 retry 行为相同，无额外退化)
```

---

### 3.3 Round 2（角色交换后）

假设场景 B 发生，角色已交换：`Primary=B, Backup=A`

```
Draft B 持有 KV cache 至 [..., d0, x1, b0_x1, b1_x1, b2_x1]
Draft B 乐观生成: [b3, b4, b5]
Draft B 同时记录各位置备选发给 A

Draft A 接收 [b0_x1, b1_x1, b2_x1] + 各位置 alternatives
Draft A 树形推理生成 6 条新备路 (以 b0~b2 的 alternatives 为入口)

Target verify([b0_x1, b1_x1, b2_x1])
... (同 Round 1 逻辑)
```

---

## 4. 角色动态交换机制

### 4.1 角色状态机

```
                    全部接受 or correction ∉ alternatives
                    ┌─────────────────────────────────────┐
                    ↓                                     │
         ┌─────────────────┐    correction ∈ alts    ┌──────────────────┐
         │  A=Primary      │ ──────────────────────→  │  A=Backup        │
         │  B=Backup       │                          │  B=Primary       │
         └─────────────────┘ ←──────────────────────  └──────────────────┘
                    ↑           correction ∈ alts         │
                    └─────────────────────────────────────┘
                    全部接受 or correction ∉ alternatives
```

### 4.2 角色交换的信息流

Target 仲裁后，向双方发送指令：

```
当 correction ∈ alternatives[i]:
  Target → 旧 Primary (新 Backup):
    { action: ROLE_SWAP_TO_BACKUP,
      new_primary_context: output_ids (到 correction token 为止),
      new_primary_draft: [backup_branch_continuation...],
      alternatives: {各位置备选} }          ← 来自新 Primary (B) 在 t2 的生成结果
      
  Target → 旧 Backup (新 Primary):
    { action: ROLE_SWAP_TO_PRIMARY,
      accepted_path: [d0, ..., d_{i-1}, correction],
      your_branch: branch_id }              ← 告知 B 是哪条分支被接受
```

旧 Backup (新 Primary) 已经在 t2 阶段完成了对应分支的 KV cache 建立，**无需重新 prefill**，直接进入 decode 状态。

### 4.3 KV Cache 状态

| 时刻 | Primary KV Cache | Backup KV Cache |
|------|-----------------|----------------|
| t2 | 延续主路 KV，乐观 decode | 树形 extend，建立 6 条分支的 KV |
| t3 (全接受) | 保留，继续 decode | 树形 KV 中仅保留对应主路分支，其余释放 |
| t3 (拒绝命中) | **角色变 Backup**：丢弃乐观部分，等待新指令 | **角色变 Primary**：保留命中分支 KV，释放其余 5 条 |
| t3 (拒绝未命中) | retry: 回滚到 correction token，重新 decode | 丢弃全部备路 KV |

---

## 5. 数据结构设计

### 5.1 扩展 `RemoteSpecRequest`

```python
@dataclass
class RemoteSpecRequest:
    # ── 现有字段 (不变) ──────────────────────────────────────
    request_id: str
    spec_cnt: int
    action: RemoteSpecAction
    spec_type: SpecType
    input_ids: Optional[List[int]]
    output_ids: Optional[List[int]]
    draft_token_ids: Optional[List[int]]
    num_draft_tokens: int
    draft_logprobs: Optional[List[float]]
    sampling_params: Optional[SamplingParams]
    
    # ── 新增字段 ─────────────────────────────────────────────
    # Primary Draft 发给 Target 和 Backup: 每个草稿位置的备选 token
    # shape: List[List[int]], len = num_draft_tokens, inner len = num_alternatives
    draft_alternatives: Optional[List[List[int]]] = None
    
    # Primary Draft 发给 Target 和 Backup: 每个备选 token 的 logprob
    # shape: List[List[float]], 与 draft_alternatives 对应
    draft_alternatives_logprobs: Optional[List[List[float]]] = None
    
    # Target 发给 Backup: 通知角色交换
    # 值: "primary" | "backup" | None (不变)
    new_role: Optional[str] = None
    
    # Target 发给新 Primary (旧 Backup): 告知哪条分支被接受
    # 格式: (rejection_position, alternative_index)
    # 例: (1, 0) 表示位置1的第0个alternative被采纳
    accepted_branch: Optional[Tuple[int, int]] = None
```

### 5.2 Target 端：`req_to_draft_token` 扩展

```python
# 现有结构:
req_to_draft_token: Dict[str, Dict[int, Optional[Tuple[List[int], List[float]]]]]
#   req_id -> spec_cnt -> (token_ids, logprobs)

# 新增: 每个请求的备选 token 缓存
req_to_alternatives: Dict[str, Dict[int, Optional[List[List[int]]]]]
#   req_id -> spec_cnt -> alternatives[pos][alt_idx]

# 新增: Backup Draft 返回的各分支续生结果
req_to_backup_branches: Dict[str, Dict[int, Optional[Dict[Tuple[int,int], List[int]]]]]
#   req_id -> spec_cnt -> {(rejection_pos, alt_idx): continuation_token_ids}

# 新增: 当前逻辑角色映射 (Target 维护)
primary_draft_identity: str    # ZMQ identity of current Primary Draft
backup_draft_identity: str     # ZMQ identity of current Backup Draft
```

### 5.3 Draft 端：`RemoteSpecDraftState` 扩展

```python
@dataclass
class RemoteSpecDraftState:
    # ── 现有字段 (不变) ──────────────────────────────────────
    req_id: str
    spec_cnt: int
    req_object: Req
    location: str
    target_origin_input_ids: List[int]
    last_prefix_length: int
    last_output_length: int
    last_updated_time: float
    
    # ── 新增字段 ─────────────────────────────────────────────
    # 当前逻辑角色
    draft_role: str = "primary"  # "primary" | "backup"
    
    # 若为 primary: 最近一轮各位置的备选 token (已发给 Target 和 Backup)
    last_alternatives: Optional[List[List[int]]] = None
    
    # 若为 backup: 当前维护的各分支 Req 对象
    # key: (rejection_pos, alt_idx), value: 该分支对应的 Req
    backup_branch_reqs: Optional[Dict[Tuple[int,int], Req]] = None
```

### 5.4 新增 `DualDraftRoleManager`（Target 端）

```python
class DualDraftRoleManager:
    """管理 Target 侧的双 Draft 角色分配与交换逻辑。"""
    
    def __init__(self):
        self.primary_identity: Optional[str] = None
        self.backup_identity: Optional[str] = None
    
    def assign_initial_roles(self, draft_identities: List[str]) -> None:
        """首次有两个 Draft 连接时分配角色。"""
        ...
    
    def should_swap_roles(
        self,
        rejection_pos: int,
        correction_token: int,
        alternatives: List[List[int]],
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """判断是否命中备路并返回命中的分支索引。"""
        ...
    
    def swap_roles(self) -> None:
        """交换 primary/backup identity。"""
        self.primary_identity, self.backup_identity = \
            self.backup_identity, self.primary_identity
```

---

## 6. 通信协议设计

### 6.1 消息流总览

```
每个 decode round 的消息流:

Phase 1: Primary → Target (现有消息 + 新增 alternatives)
  RemoteSpecRequest {
    action: DRAFT,
    spec_type: DRAFT_RESPONSE,
    draft_token_ids: [d0, d1, d2],
    draft_alternatives: [[x0,y0], [x1,y1], [x2,y2]],  ← 新增
    draft_alternatives_logprobs: [[p0x,p0y], ...],     ← 新增
  }

Phase 2: Target → Primary (现有消息，不变)
  RemoteSpecRequest {
    action: DRAFT,
    spec_type: DRAFT_REQUEST,
    output_ids: [..., d0, d1, d2, bonus],
    draft_token_ids: [d0, d1, d2],   ← cur_drafts for fork-point
    num_draft_tokens: N,
  }

Phase 2: Target → Backup (新增消息)
  RemoteSpecRequest {
    action: DRAFT,
    spec_type: DRAFT_REQUEST,
    output_ids: [...],                   ← context (不含主路草稿)
    draft_token_ids: [d0, d1, d2],       ← 主路草稿 (Backup 用于构建树)
    draft_alternatives: [[x0,y0], ...],  ← 各位置备选
    num_draft_tokens: N,
    new_role: "backup",                  ← 明确告知是备路任务
  }

Phase 3: Backup → Target (新增消息，与 Primary 消息并行)
  RemoteSpecRequest {
    action: DRAFT,
    spec_type: DRAFT_RESPONSE,
    draft_token_ids: [],                 ← 备路不发主路 token
    backup_branches: {                   ← 新增: 所有分支的续生
      (0, 0): [b00, b01, b02],           ← 位置0,alt0 (x0) 的续生
      (0, 1): [b01, b11, b21],           ← 位置0,alt1 (y0) 的续生
      (1, 0): [b10, b11, b12],
      ...
    }
  }

Phase 4 (角色交换时): Target → 双方 (新增消息)
  → 旧 Primary (新 Backup):
    { action: ROLE_SWAP, new_role: "backup",
      new_primary_draft: [correction, branch_continuation...],
      draft_alternatives: {...} }   ← 来自新 Primary 在 t2 的生成
      
  → 旧 Backup (新 Primary):
    { action: ROLE_SWAP, new_role: "primary",
      accepted_branch: (rejection_pos, alt_idx) }
```

### 6.2 扩展 `RemoteSpecAction`

```python
class RemoteSpecAction(str, Enum):
    DRAFT   = "draft"     # 现有
    FINISH  = "finish"    # 现有
    ABORT   = "abort"     # 现有
    REJECT  = "reject"    # 现有
    ROLE_SWAP = "role_swap"  # 新增: 通知角色交换
```

### 6.3 `_zmq_send` 路由逻辑扩展

```python
def _zmq_send_primary(self, reqs: List[RemoteSpecRequest]) -> None:
    """发送到当前 Primary Draft。"""
    self.zmq_communicator.send_objs(reqs, self.role_manager.primary_identity)

def _zmq_send_backup(self, reqs: List[RemoteSpecRequest]) -> None:
    """发送到当前 Backup Draft。"""
    if self.role_manager.backup_identity:
        self.zmq_communicator.send_objs(reqs, self.role_manager.backup_identity)

def _zmq_send_both(
    self,
    primary_reqs: List[RemoteSpecRequest],
    backup_reqs: List[RemoteSpecRequest],
) -> None:
    """同时向 Primary 和 Backup 发送（不同内容）。"""
    self._zmq_send_primary(primary_reqs)
    self._zmq_send_backup(backup_reqs)
```

---

## 7. 各模块实现方案

### 7.1 Primary Draft 端（`remote_spec_draft_scheduler_mixin.py`）

#### 7.1.1 `_send_draft_response` — 附带 alternatives

```python
def _send_draft_response(self, req: Req) -> None:
    draft_tokens = req.output_ids[req.draft_generation_start_len:]
    
    # 现有: 收集 logprobs
    draft_logits = self._collect_draft_logprobs(req, draft_tokens)
    
    # 新增: 收集各位置的 top-k alternatives
    # req.output_top_logprobs_val / idx 已包含 top-k 信息 (return_logprob=True, top_logprobs_num=k)
    alternatives, alt_logprobs = self._collect_alternatives(req, draft_tokens)
    
    response = RemoteSpecRequest(
        ...
        draft_token_ids=draft_tokens,
        draft_logprobs=draft_logits,
        draft_alternatives=alternatives,        # 新增
        draft_alternatives_logprobs=alt_logprobs,  # 新增
    )
    self.zmq_communicator.send_objs([response])

def _collect_alternatives(
    self, req: Req, draft_tokens: List[int]
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    从 req.output_top_logprobs_val/idx 中提取各位置 top-k alternatives。
    
    前提: 创建 Draft Req 时已设置 top_logprobs_num >= num_alternatives + 1
    (+1 是因为 top-1 就是 primary token 本身)
    """
    start = req.draft_generation_start_len
    alternatives, alt_logprobs = [], []
    
    for i in range(len(draft_tokens)):
        pos = start + i
        if (req.output_top_logprobs_idx and 
            pos < len(req.output_top_logprobs_idx)):
            top_indices = req.output_top_logprobs_idx[pos]   # List[int], len=k
            top_values  = req.output_top_logprobs_val[pos]   # List[float], len=k
            # 跳过 top-1 (= primary token)，取 [1:]
            alts = [idx for idx in top_indices[1:]]
            alt_lp = list(top_values[1:])
        else:
            alts, alt_lp = [], []
        alternatives.append(alts)
        alt_logprobs.append(alt_lp)
    
    return alternatives, alt_logprobs
```

#### 7.1.2 `_create_new_draft_req` — 设置 `top_logprobs_num`

```python
# 修改: return_logprob=True, top_logprobs_num = num_alternatives + 1
req = Req(
    ...
    return_logprob=True,
    top_logprobs_num=self.server_args.remote_speculative_num_alternatives + 1,
    ...
)
```

#### 7.1.3 Backup Draft 的树形推理（新增逻辑）

Backup Draft 收到 `new_role="backup"` 的请求后，走不同的处理分支：

```python
def _process_backup_request(
    self, draft_req: RemoteSpecRequest
) -> None:
    """
    处理备路任务: 对每个 (rejection_pos, alternative) 组合构建独立 Req，
    并行 prefill + 生成续生 token。
    
    实现策略 (简化版): 
      每条分支对应一个独立的 Req 对象加入 waiting_queue，
      填充内容 = context + [d0,...,d_{i-1}] + [alt] + [d_{i+1},...,d_{N-1}]
      让 Scheduler 统一调度 prefill。
      所有分支 Req 完成 prefill 后各自 decode 1 步，得到续生 token。
    
    优化方向 (后续迭代):
      使用树形 attention (EXTEND mode + custom_mask) 一次 forward 覆盖所有分支，
      避免多次 prefill 的 overhead。
    """
    main_draft = draft_req.draft_token_ids or []
    alternatives = draft_req.draft_alternatives or []
    context = (draft_req.input_ids or []) + (draft_req.output_ids or [])
    num_alternatives = len(alternatives[0]) if alternatives else 0
    
    branch_reqs = {}
    for pos in range(len(main_draft)):
        for alt_idx, alt_token in enumerate(alternatives[pos] if pos < len(alternatives) else []):
            # 构建分支的 fill_ids:
            # context + main_draft[0:pos] + alt_token + main_draft[pos+1:]
            branch_ids = (
                context
                + main_draft[:pos]
                + [alt_token]
                + main_draft[pos+1:]
            )
            branch_req = self._create_branch_req(
                base_req_id=draft_req.request_id,
                branch_key=(pos, alt_idx),
                fill_ids=branch_ids,
                num_new_tokens=self.server_args.speculative_num_steps,
                sampling_params=draft_req.sampling_params,
            )
            branch_reqs[(pos, alt_idx)] = branch_req
            self._add_request_to_queue(branch_req)
    
    # 记录 branch_reqs 到 state，等待所有分支完成后汇总发送
    state = self._get_draft_state(draft_req.request_id)
    if state:
        state.backup_branch_reqs = branch_reqs
```

---

### 7.2 Target 端调度（`remote_spec_target_scheduler_mixin.py`）

#### 7.2.1 `send_batch_draft_requests` — 同时向 Primary 和 Backup 发送

```python
def send_batch_draft_requests(
    self, batch: ScheduleBatch, speculative_num_draft_tokens: int
) -> None:
    """向 Primary 发主路请求，向 Backup 发备路构建请求。"""
    primary_reqs, backup_reqs = [], []
    
    for req in batch.reqs:
        if _is_health_check(req):
            continue
        self.req_to_draft_token[req.rid][req.spec_cnt] = None
        
        # 主路请求 (与现有基本相同)
        primary_reqs.append(RemoteSpecRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=RemoteSpecAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=req.origin_input_ids if req.spec_cnt == 0 else None,
            output_ids=req.output_ids,
            draft_token_ids=req.cur_drafts,
            num_draft_tokens=speculative_num_draft_tokens,
            sampling_params=req.sampling_params if req.spec_cnt == 0 else None,
        ))
        
        # 备路请求 (仅当 alternatives 可用时，即 spec_cnt > 0)
        alts = self.req_to_alternatives.get(req.rid, {}).get(req.spec_cnt - 1)
        if alts and self.role_manager.backup_identity:
            backup_reqs.append(RemoteSpecRequest(
                request_id=req.rid,
                spec_cnt=req.spec_cnt,
                action=RemoteSpecAction.DRAFT,
                spec_type=SpecType.DRAFT_REQUEST,
                output_ids=req.output_ids,
                draft_token_ids=req.cur_drafts,         # 主路草稿
                draft_alternatives=alts,                 # 各位置备选
                num_draft_tokens=speculative_num_draft_tokens,
                new_role="backup",                       # 明确是备路任务
            ))
    
    if self.tp_size == 1 or self.tp_rank == 0:
        self._zmq_send_primary(primary_reqs)
        if backup_reqs:
            self._zmq_send_backup(backup_reqs)
```

#### 7.2.2 `_store_messages` — 分别缓存 Primary 和 Backup 响应

```python
def _store_messages(self, messages: List[Tuple[str, RemoteSpecRequest]]) -> bool:
    has_draft = False
    for draft_id, msg in messages:
        if msg.action == RemoteSpecAction.REJECT:
            self.process_reject_action()
        elif msg.action == RemoteSpecAction.DRAFT:
            if draft_id == self.role_manager.primary_identity:
                # Primary 的主路 token + alternatives
                self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                    msg.draft_token_ids, msg.draft_logprobs
                )
                if msg.draft_alternatives:
                    self.req_to_alternatives[msg.request_id][msg.spec_cnt] = (
                        msg.draft_alternatives
                    )
                has_draft = True
            elif draft_id == self.role_manager.backup_identity:
                # Backup 的各分支续生结果
                if hasattr(msg, 'backup_branches') and msg.backup_branches:
                    self.req_to_backup_branches[msg.request_id][msg.spec_cnt] = (
                        msg.backup_branches
                    )
    return has_draft
```

#### 7.2.3 `_post_verify_check_backup` — verify 后检查备路

在 `RemoteSpecWorker._post_verify_update_drafts` 的拒绝路径中插入备路检查：

```python
def _check_and_use_backup_branch(
    self,
    req: Req,
    rejection_pos: int,
    correction_token: int,
) -> bool:
    """
    尝试用备路续生序列直接替换 retry。
    
    返回 True 表示命中备路，req 的 draft state 已更新，无需 retry。
    返回 False 表示未命中，调用方走 retry 流程。
    """
    alts = self.req_to_alternatives.get(req.rid, {}).get(req.spec_cnt - 1)
    backup_branches = self.req_to_backup_branches.get(req.rid, {}).get(req.spec_cnt - 1)
    
    if not alts or not backup_branches:
        return False
    
    # 检查 correction_token 是否在 rejection_pos 的 alternatives 中
    if rejection_pos >= len(alts):
        return False
    
    for alt_idx, alt_token in enumerate(alts[rejection_pos]):
        if alt_token == correction_token:
            branch_key = (rejection_pos, alt_idx)
            continuation = backup_branches.get(branch_key)
            if continuation:
                # 命中! 使用备路续生序列
                req.cur_drafts = list(continuation)
                req.draft_tokens_and_logits = _make_draft_dict(
                    continuation, [0.0] * len(continuation)
                )
                # 通知角色交换
                self._trigger_role_swap(req, rejection_pos, alt_idx)
                return True
    
    return False

def _trigger_role_swap(
    self,
    req: Req,
    rejection_pos: int,
    alt_idx: int,
) -> None:
    """发送角色交换通知，更新本地角色映射。"""
    # 通知旧 Backup (新 Primary)
    swap_to_primary = RemoteSpecRequest(
        request_id=req.rid,
        spec_cnt=req.spec_cnt,
        action=RemoteSpecAction.ROLE_SWAP,
        accepted_branch=(rejection_pos, alt_idx),
        new_role="primary",
    )
    self._zmq_send_backup([swap_to_primary])
    
    # 通知旧 Primary (新 Backup): 携带新主路信息供 B 下轮使用
    swap_to_backup = RemoteSpecRequest(
        request_id=req.rid,
        spec_cnt=req.spec_cnt,
        action=RemoteSpecAction.ROLE_SWAP,
        output_ids=req.output_ids,
        draft_token_ids=req.cur_drafts,   # = backup continuation
        new_role="backup",
    )
    self._zmq_send_primary([swap_to_backup])
    
    # 更新本地角色映射
    self.role_manager.swap_roles()
```

---

### 7.3 `RemoteSpecWorker` 端（`remote_spec_worker.py`）

#### 7.3.1 修改 `_post_verify_update_drafts`

```python
def _post_verify_update_drafts(self, batch, res, new_drafts_per_req, ...):
    failed_reqs = []

    for i, req in enumerate(batch.reqs):
        if _is_health_check(req):
            continue

        drafts = new_drafts_per_req.get(req.rid)

        if drafts is not None:
            draft_token_ids, draft_logprobs = drafts
            verified_tokens = req.output_ids[req.len_output_ids:]
            cur_draft_tokens = list(getattr(req, "cur_drafts", []))
            cur_draft_tokens.append(draft_token_ids[0])

            is_matched, matched_idx = _find_fork_point(verified_tokens, cur_draft_tokens)
            req.draft_cnt += len(cur_draft_tokens)
            req.accept_cnt += matched_idx

            if is_matched:
                # 现有 pipelined path (不变)
                req.cur_drafts = list(draft_token_ids[1:])
                req.draft_tokens_and_logits = _make_draft_dict(
                    draft_token_ids[1:], draft_logprobs[1:]
                )
            else:
                # 新增: 尝试从 Backup 备路中找到 correction token
                rejection_pos = matched_idx  # 第一个不匹配的位置
                correction_token = verified_tokens[rejection_pos] if rejection_pos < len(verified_tokens) else -1
                
                backup_hit = self._check_and_use_backup_branch(
                    req, rejection_pos, correction_token
                )
                
                if not backup_hit:
                    req.cur_drafts = []
                    req.draft_tokens_and_logits = _default_draft()
                    failed_reqs.append(req)
                # 若 backup_hit=True, req.cur_drafts 已由备路填充，无需 retry
        else:
            req.cur_drafts = []
            req.draft_tokens_and_logits = _default_draft()
            failed_reqs.append(req)

        req.spec_cnt += 1
        req.len_output_ids = len(req.output_ids)

    # retry 逻辑 (仅对未命中备路的请求)
    # ... (与现有相同)
```

---

## 8. 边界情况与容错

### 8.1 只有一个 Draft 连接

```
条件: role_manager.backup_identity is None
处理: 退化为现有 1:1 行为，不发送备路请求，不尝试备路检查
```

### 8.2 Backup Draft 超时未返回

```
条件: t3 时 req_to_backup_branches[rid][spec_cnt] 为空
处理: _check_and_use_backup_branch 返回 False
      走现有 retry 流程，不影响正确性
      记录统计: backup_miss_count++
```

### 8.3 角色交换通知丢失

```
条件: ROLE_SWAP 消息因网络问题丢失
处理: 各 Draft 端收到下一轮请求时，根据 new_role 字段重新对齐角色
      Draft 端维护 "当前 spec_cnt 对应的角色" 状态
      超时未收到角色确认: 保持旧角色直至下一轮重新同步
```

### 8.4 角色交换时的 KV Cache 清理

```
旧 Primary (新 Backup):
  收到 ROLE_SWAP 后:
  1. 将乐观续生部分 (draft_generation_start_len 之后) 的 KV rollback
  2. 等待新的备路任务请求 (包含新主路信息)

旧 Backup (新 Primary):
  收到 accepted_branch=(pos, alt_idx) 后:
  1. 保留对应分支的 KV cache (该分支在 t2 extend 时已建立)
  2. 释放其余 (num_pos * num_alt - 1) 条分支的 KV
  3. 进入 decode 状态，继续乐观生成
```

### 8.5 Backup Draft 过载

```
条件: Backup Draft 返回 REJECT 消息
处理: Target 停止向 Backup 发送请求，退化为 1:1
      复用现有 DraftCircuitBreaker 逻辑，但针对 Backup 独立维护一个 breaker 实例
```

---

## 9. 性能分析

### 9.1 理论收益

设：
- `α` = 单轮全部接受概率
- `β_i` = 在位置 `i` 拒绝且 correction token ∈ alternatives 的概率
- `k` = num_alternatives (每位置备选数)
- `T_draft` = Draft 重新生成一批草稿的 RTT

**现有 1:1**：
```
E[latency_penalty] = (1 - α) × T_draft
```

**新 1:2**：
```
E[latency_penalty] = (1 - α - Σβ_i) × T_draft
                   ≈ (1 - α) × (1 - hit_rate) × T_draft

其中 hit_rate = P(correction ∈ top-k alternatives)
```

**`hit_rate` 的理论估计**（基于 draft 和 target 分布对齐程度）：
- top-1 命中率 ≈ accept rate (α)
- top-2 额外命中率 ≈ 0.1~0.2（通常第二高概率 token 有较高价值）
- top-3 额外命中率 ≈ 0.05~0.1
- k=2 时，综合 hit_rate 估计约 0.15~0.30（在 reject 事件中）

### 9.2 额外开销

| 开销项 | 量化 |
|-------|------|
| Backup Draft 树形推理 | 约 1 次 Primary forward 等量（但与 Primary 并行，不增加墙上时间）|
| 网络传输增量 | 每轮多传 `num_draft_tokens × num_alt × 4 bytes` alternatives（< 1KB） |
| Target 端备路检查 | O(num_draft_tokens × num_alt) Python 比较，< 0.1ms |
| KV Cache 占用 | Backup 额外占用 `num_pos × num_alt × page_size` 个 slot（可控） |

### 9.3 参数建议

| 参数 | 推荐默认值 | 说明 |
|------|----------|------|
| `num_alternatives` (k) | 2 | 覆盖 top-3，收益边际递减快 |
| Backup 分支数 | `num_draft_tokens × k` | 当前例中 = 3×2=6 |
| Backup Req top_logprobs_num | k+1 | Backup 自身也记录备选供下轮使用 |

---

## 附录：实现优先级建议

**Phase 1（基础框架，验证核心收益）**：
1. Primary Draft `_send_draft_response` 附带 alternatives（修改点最小）
2. Target 端 `req_to_alternatives` 缓存
3. Backup Draft 接收备路任务，串行构建分支 Req（简化实现）
4. Target 端拒绝时查找备路（`_check_and_use_backup_branch`）
5. 统计 backup_hit_rate，验证收益

**Phase 2（角色交换优化）**：
1. ROLE_SWAP 消息 + `DualDraftRoleManager`
2. 旧 Primary 收到 ROLE_SWAP 后的 KV rollback
3. 旧 Backup 收到 ROLE_SWAP 后保留命中分支 KV

**Phase 3（树形推理优化）**：
1. Backup Draft 改用树形 attention 一次 forward 覆盖所有分支（需要 custom_mask 支持）
2. 减少 Backup 的 prefill 重复计算（共享 prefix KV）
