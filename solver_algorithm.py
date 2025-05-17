# solver_algorithm.py
import numpy as np
import heapq
from collections import deque
import time

# --- Lớp Node để lưu trữ trạng thái ---
class Node:
    def __init__(self, board, parent=None, action=None, g_cost=0):
        self.board = board # Ma trận numpy
        self.parent = parent
        self.action = action # Hành động để đạt được trạng thái này (UP, DOWN, LEFT, RIGHT)
        self.g_cost = g_cost # Chi phí từ trạng thái bắt đầu
        self.h_cost = 0 # Chi phí heuristic (sẽ được tính riêng)
        self.f_cost = 0 # f_cost = g_cost + h_cost (cho A* và IDA*)

    def __lt__(self, other):
        if self.f_cost == other.f_cost:
            return self.g_cost < other.g_cost
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        # So sánh chỉ board để dùng trong set và dict
        if isinstance(other, Node):
            return np.array_equal(self.board, other.board)
        return False


    def __hash__(self):
        return hash(self.board.tobytes())

    def get_board_tuple(self):
        return tuple(map(tuple, self.board.tolist()))

# --- Hàm tiện ích ---
def get_goal_state(n):
    goal_list = list(range(n * n))
    goal_np = np.array(goal_list).reshape((n, n))
    return goal_np

def find_blank(board):
    pos = np.where(board == 0)
    return pos[0][0], pos[1][0]

def get_possible_moves(board):
    moves = []
    r, c = find_blank(board)
    n = board.shape[0]
    directions = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
    for dr, dc, action_name in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            new_board = board.copy()
            new_board[r, c], new_board[nr, nc] = new_board[nr, nc], new_board[r, c]
            moves.append((new_board, action_name))
    return moves

def reconstruct_path(node):
    path = []
    current = node
    while current.parent:
        path.append(current.action)
        current = current.parent
    return path[::-1]

# --- Heuristics cho A* và IDA* ---
def manhattan_distance(board, goal_state):
    distance = 0
    n = board.shape[0]
    for i in range(n):
        for j in range(n):
            tile_value = board[i, j]
            if tile_value != 0:
                goal_pos = np.where(goal_state == tile_value)
                goal_r, goal_c = goal_pos[0][0], goal_pos[1][0]
                distance += abs(i - goal_r) + abs(j - goal_c)
    return distance

# --- Các thuật toán tìm kiếm ---
def bfs(initial_board_np, goal_state_np):
    start_time = time.time()
    initial_node = Node(initial_board_np)
    if np.array_equal(initial_node.board, goal_state_np):
        return [], 0, time.time() - start_time

    queue = deque([initial_node])
    visited = {initial_node.get_board_tuple()}
    nodes_expanded = 0

    while queue:
        current_node = queue.popleft()
        nodes_expanded += 1
        for next_board, action in get_possible_moves(current_node.board):
            next_node = Node(next_board, parent=current_node, action=action, g_cost=current_node.g_cost + 1)
            if next_node.get_board_tuple() not in visited:
                if np.array_equal(next_node.board, goal_state_np):
                    path = reconstruct_path(next_node)
                    return path, nodes_expanded, time.time() - start_time
                visited.add(next_node.get_board_tuple())
                queue.append(next_node)
    return None, nodes_expanded, time.time() - start_time

def dfs(initial_board_np, goal_state_np):
    start_time = time.time()
    initial_node = Node(initial_board_np)
    
    if np.array_equal(initial_node.board, goal_state_np):
        return [], 0, time.time() - start_time

    stack = [initial_node] 
    visited = {initial_node.get_board_tuple()}
    nodes_expanded = 0

    while stack:
        current_node = stack.pop()
        nodes_expanded += 1

        if np.array_equal(current_node.board, goal_state_np):
            path = reconstruct_path(current_node)
            return path, nodes_expanded, time.time() - start_time

        for next_board, action in reversed(get_possible_moves(current_node.board)):
            next_board_tuple = tuple(map(tuple, next_board.tolist()))
            if next_board_tuple not in visited: 
                next_node = Node(next_board, parent=current_node, action=action)
                visited.add(next_board_tuple)
                stack.append(next_node)
    return None, nodes_expanded, time.time() - start_time


def a_star(initial_board_np, goal_state_np, heuristic_func=manhattan_distance):
    start_time = time.time()
    initial_node = Node(initial_board_np, g_cost=0)
    initial_node.h_cost = heuristic_func(initial_node.board, goal_state_np)
    initial_node.f_cost = initial_node.g_cost + initial_node.h_cost

    if np.array_equal(initial_node.board, goal_state_np):
        return [], 0, time.time() - start_time

    open_set = [initial_node]
    # Sử dụng g_costs_map để lưu trữ g_cost nhỏ nhất tới một board_tuple
    # và dùng nó để kiểm tra visited thay cho closed_set dạng set(node)
    # Điều này tránh vấn đề hash Node object trực tiếp nếu nó có thuộc tính thay đổi hoặc
    # cho phép cập nhật node trong open_set (mặc dù heapq không hỗ trợ update trực tiếp,
    # nên chiến lược thường là push node mới và bỏ qua node cũ khi pop)
    g_costs_map = {initial_node.get_board_tuple(): 0}
    nodes_expanded = 0

    while open_set:
        current_node = heapq.heappop(open_set)
        nodes_expanded += 1

        # Nếu node này đã được xử lý với g_cost tốt hơn (hoặc bằng), bỏ qua
        if current_node.g_cost > g_costs_map.get(current_node.get_board_tuple(), float('inf')):
            continue

        if np.array_equal(current_node.board, goal_state_np):
            path = reconstruct_path(current_node)
            return path, nodes_expanded, time.time() - start_time

        for next_board, action in get_possible_moves(current_node.board):
            next_g_cost = current_node.g_cost + 1
            next_board_tuple = tuple(map(tuple, next_board.tolist()))

            if next_g_cost < g_costs_map.get(next_board_tuple, float('inf')):
                g_costs_map[next_board_tuple] = next_g_cost
                next_node = Node(next_board, parent=current_node, action=action, g_cost=next_g_cost)
                next_node.h_cost = heuristic_func(next_node.board, goal_state_np)
                next_node.f_cost = next_node.g_cost + next_node.h_cost
                heapq.heappush(open_set, next_node)
    return None, nodes_expanded, time.time() - start_time

def ida_star_search(current_node, goal_state_np, g_cost, bound, heuristic_func, path_actions):
    nodes_expanded_this_iteration = 1
    f_cost = g_cost + heuristic_func(current_node.board, goal_state_np)

    if f_cost > bound:
        return f_cost, None, nodes_expanded_this_iteration
    if np.array_equal(current_node.board, goal_state_np):
        return "FOUND", path_actions, nodes_expanded_this_iteration

    min_next_bound = float('inf')
    
    for next_b, act in get_possible_moves(current_node.board):
        if current_node.parent and np.array_equal(next_b, current_node.parent.board):
            continue
        
        next_node_for_recursion = Node(next_b, parent=current_node, action=act) 

        result, found_path, expanded_children = ida_star_search(
            next_node_for_recursion, goal_state_np, g_cost + 1, bound, heuristic_func, path_actions + [act]
        )
        nodes_expanded_this_iteration += expanded_children
        if result == "FOUND":
            return "FOUND", found_path, nodes_expanded_this_iteration
        if result < min_next_bound:
            min_next_bound = result
    return min_next_bound, None, nodes_expanded_this_iteration

def ida_star(initial_board_np, goal_state_np, heuristic_func=manhattan_distance):
    start_time = time.time()
    initial_node = Node(initial_board_np, g_cost=0)

    if np.array_equal(initial_node.board, goal_state_np):
        return [], 0, 0, time.time() - start_time

    bound = heuristic_func(initial_node.board, goal_state_np)
    total_nodes_expanded = 0
    iterations = 0

    while True:
        iterations += 1
        temp_initial_node = Node(initial_board_np.copy()) # Đảm bảo board không bị thay đổi giữa các lần lặp
        result, path, nodes_in_iter = ida_star_search(
            temp_initial_node, goal_state_np, 0, bound, heuristic_func, []
        )
        total_nodes_expanded += nodes_in_iter

        if result == "FOUND":
            return path, total_nodes_expanded, iterations, time.time() - start_time
        if result == float('inf'):
            return None, total_nodes_expanded, iterations, time.time() - start_time
        bound = result

# --- Kiểm tra tính giải được ---
def get_inversions(board_flat):
    inversions = 0
    n_flat = len(board_flat)
    for i in range(n_flat):
        for j in range(i + 1, n_flat):
            if board_flat[i] != 0 and board_flat[j] != 0 and board_flat[i] > board_flat[j]:
                inversions += 1
    return inversions

def is_solvable(board_np): # Nhận board numpy
    n = board_np.shape[0]
    flat_board = board_np.flatten()
    inversions = get_inversions(flat_board)
    blank_row, _ = find_blank(board_np)
    if n % 2 == 1:
        return inversions % 2 == 0
    else:
        blank_row_from_bottom = (n - 1) - blank_row
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0