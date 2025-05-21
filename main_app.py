# main_app.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json # Mặc dù không lưu vào board.json nữa, nhưng có thể cần cho tương lai
import numpy as np
import solver_algorithm as solver # Import module thuật toán

class NPuzzleGUI:
    def __init__(self, master):
        self.master = master
        master.title("N-Puzzle Solver")
        master.geometry("800x750")

        self.current_n = 3
        self.entry_widgets = []
        self.initial_board_np = None

        # --- Top Frame: Size Options ---
        self.size_frame = ttk.Frame(master, padding="10")
        self.size_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.size_frame, text="Select Puzzle Size:").pack(side=tk.LEFT, padx=5)
        self.size_var = tk.IntVar(value=self.current_n)
        for size in [3, 4, 5]: # Giới hạn kích thước
            rb = ttk.Radiobutton(
                self.size_frame,
                text=f"{size}x{size}",
                variable=self.size_var,
                value=size,
                command=self.change_size
            )
            rb.pack(side=tk.LEFT, padx=5)

        # --- Middle Frame: Input Grid and Remaining Numbers ---
        self.input_area_frame = ttk.Frame(master, padding="10")
        self.input_area_frame.pack(expand=False, fill=tk.X, pady=5) # expand=False để không chiếm quá nhiều không gian dọc

        # Frame cho lưới nhập liệu
        self.matrix_frame_container = ttk.Frame(self.input_area_frame)
        self.matrix_frame_container.pack(side=tk.LEFT, padx=(0, 20), anchor="nw")
        self.matrix_frame = ttk.Frame(self.matrix_frame_container)
        self.matrix_frame.pack()

        # Frame cho hiển thị số còn lại
        self.remaining_numbers_frame = ttk.Frame(self.input_area_frame, padding="5")
        self.remaining_numbers_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, anchor="ne") # fill Y, no expand X

        ttk.Label(self.remaining_numbers_frame, text="Numbers Guide:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.remaining_numbers_label_var = tk.StringVar()
        self.remaining_numbers_label = ttk.Label(
            self.remaining_numbers_frame,
            textvariable=self.remaining_numbers_label_var,
            wraplength=200,
            justify=tk.LEFT,
            font=("Arial", 9)
        )
        self.remaining_numbers_label.pack(anchor="nw", pady=5, fill=tk.X)

        # --- Solver Controls Frame ---
        self.solver_controls_frame = ttk.Frame(master, padding="10")
        self.solver_controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.solver_controls_frame, text="Choose Algorithm:").pack(side=tk.LEFT, padx=(0,10))
        self.selected_algorithm = tk.StringVar(value="BFS") # Giá trị mặc định
        algo_options = ["BFS", "DFS", "A* (Manhattan)", "IDA* (Manhattan)"]
        self.algo_combobox = ttk.Combobox(self.solver_controls_frame, textvariable=self.selected_algorithm, values=algo_options, state="readonly", width=18)
        self.algo_combobox.pack(side=tk.LEFT, padx=5)

        # --- Action Buttons (Solve, Clear) ---
        self.action_buttons_frame = ttk.Frame(master, padding="10") # Đổi tên để rõ ràng hơn
        self.action_buttons_frame.pack(fill=tk.X, pady=5)

        self.solve_button = ttk.Button(self.action_buttons_frame, text="Solve Puzzle", command=self.initiate_solve_process)
        self.solve_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.clear_button = ttk.Button(self.action_buttons_frame, text="Clear Board", command=self.clear_board_entries)
        self.clear_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        # --- Status Label ---
        self.status_label_var = tk.StringVar()
        self.status_label = ttk.Label(master, textvariable=self.status_label_var, padding="5", foreground="blue", wraplength=780)
        self.status_label.pack(fill=tk.X, pady=(0,5))

        # --- Results Display ---
        self.results_frame = ttk.LabelFrame(master, text="Solver Results", padding="10")
        self.results_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=(5,10))

        self.result_text_area = scrolledtext.ScrolledText(self.results_frame, height=10, wrap=tk.WORD, font=("Arial", 10))
        self.result_text_area.pack(expand=True, fill=tk.BOTH)
        self.result_text_area.config(state=tk.DISABLED)

        self.change_size()

    def _get_all_possible_numbers(self):
        return set(range(self.current_n * self.current_n))

    def _update_remaining_numbers_display(self):
        all_nums = self._get_all_possible_numbers()
        entered_nums = set()
        for r_idx in range(len(self.entry_widgets)):
            for c_idx in range(len(self.entry_widgets[r_idx])):
                val = getattr(self.entry_widgets[r_idx][c_idx], 'last_valid_value', None)
                if val is not None:
                    entered_nums.add(val)

        remaining = sorted(list(all_nums - entered_nums))
        expected_count = self.current_n * self.current_n

        if not remaining and len(entered_nums) == expected_count:
            text = "All numbers used correctly.\nReady to solve!"
            color = "green"
        elif len(entered_nums) < expected_count :
            text = f"Numbers to fill (0 to {expected_count-1}):\n" + \
                   (", ".join(map(str, remaining)) if remaining else "None left")
            color = "black"
        else:
            text = "Check input. Possible duplicates or out-of-range."
            color = "red"
        self.remaining_numbers_label_var.set(text)
        self.remaining_numbers_label.config(foreground=color)

    def _clear_status(self):
        self.status_label_var.set("")
        self.status_label.config(foreground="blue")

    def _set_status(self, message, color="blue"):
        self.status_label.config(foreground=color)
        self.status_label_var.set(message)

    def _check_data_in_board(self, board):
        """
        Kiểm tra bảng đầu vào:
        - Nếu có ô trùng lặp, trả về ('duplicate', [(r1, c1), (r2, c2)], value)
        - Nếu có ô ngoài khoảng, trả về ('out_of_range', (r, c), value)
        - Nếu hợp lệ, trả về None
        """
        n = len(board)
        seen = {}
        max_val = n * n - 1

        # Kiểm tra trùng lặp và ngoài khoảng
        for r in range(n):
            for c in range(n):
                val = board[r][c]
                # Kiểm tra ngoài khoảng
                if not (0 <= val <= max_val):
                    return ('out_of_range', (r, c), val)
                # Kiểm tra trùng lặp
                if val in seen:
                    return ('duplicate', [seen[val], (r, c)], val)
                seen[val] = (r, c)
        return None

    def create_grid(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        self.entry_widgets = []
        entry_width = 3
        font_size = 18 if self.current_n <= 4 else (16 if self.current_n <= 5 else 14)

        for r in range(self.current_n):
            row_entries = []
            for c in range(self.current_n):
                entry_var = tk.StringVar()
                entry = ttk.Entry(
                    self.matrix_frame, textvariable=entry_var, width=entry_width,
                    font=('Arial', font_size), justify='center'
                )
                entry.grid(row=r, column=c, padx=2, pady=2, ipady=5)
                setattr(entry, 'last_valid_value', None)
                entry_var.trace_add("write", lambda name, index, mode, sv=entry_var, r_idx=r, c_idx=c: \
                                    self._validate_and_handle_entry(name, index, mode, r_idx, c_idx))
                row_entries.append(entry)
            self.entry_widgets.append(row_entries)

    def change_size(self):
        new_n = self.size_var.get()
        if new_n != self.current_n or not self.entry_widgets: # Also recreate if no widgets (initial)
            self.current_n = new_n
            self.clear_board_entries(called_from_change_size=True) # Pass a flag
            self.create_grid()
        self._update_remaining_numbers_display()
        self._clear_status()
        self.result_text_area.config(state=tk.NORMAL)
        self.result_text_area.delete(1.0, tk.END)
        self.result_text_area.config(state=tk.DISABLED)

    def clear_board_entries(self, called_from_change_size=False):
        for r_list in self.entry_widgets:
            for entry_widget in r_list:
                var_name = entry_widget.cget("textvariable")
                if var_name: # Get the StringVar object
                    string_var = self.master.getvar(var_name)
                    # Find and remove trace temporarily - simpler to just clear value
                    # This is complex to do correctly, direct deletion is easier
                    entry_widget.delete(0, tk.END) # This will trigger trace, but last_valid_value helps
                setattr(entry_widget, 'last_valid_value', None)
        
        self.initial_board_np = None
        if not called_from_change_size: # Avoid re-updating if change_size will do it
            self._update_remaining_numbers_display()
            self._clear_status()
            self.result_text_area.config(state=tk.NORMAL)
            self.result_text_area.delete(1.0, tk.END)
            self.result_text_area.config(state=tk.DISABLED)


    def _collect_board_data_for_solving(self):
        """Collects data from UI, validates, and returns np.array or None."""
        board_data_list = []
        seen_numbers = set()
        max_val = self.current_n * self.current_n - 1

        for r in range(self.current_n):
            row_data = []
            for c in range(self.current_n):
                entry_widget = self.entry_widgets[r][c]
                val_str = entry_widget.get()
                
                if not val_str:
                    self._set_status(f"Cell ({r+1},{c+1}) is empty. Fill all cells.", "red")
                    return None

                if not val_str.isdigit():
                    self._set_status(f"Cell ({r+1},{c+1}): '{val_str}' is not a number.", "red")
                    return None

                val_int = int(val_str)

                if not (0 <= val_int <= max_val):
                    self._set_status(f"Cell ({r+1},{c+1}): {val_int} out of range [0, {max_val}].", "red")
                    return None

                if val_int in seen_numbers:
                    self._set_status(f"Number {val_int} is duplicated. Use unique numbers.", "red")
                    return None
                
                seen_numbers.add(val_int)
                row_data.append(val_int)
            board_data_list.append(row_data)

        if len(seen_numbers) != self.current_n * self.current_n:
            self._set_status(f"Not all numbers from 0 to {max_val} are used.", "red")
            return None

        self._set_status("Board data is valid and ready for solving.", "green")
        return np.array(board_data_list)


    def initiate_solve_process(self):
        self._clear_status()
        self.result_text_area.config(state=tk.NORMAL)
        self.result_text_area.delete(1.0, tk.END)

        current_board_np = self._collect_board_data_for_solving()
        if current_board_np is None:
            self.result_text_area.insert(tk.END, "Board input is invalid. Cannot solve.\nFix errors shown in the status bar.\n")
            self.result_text_area.config(state=tk.DISABLED)
            return

        self.initial_board_np = current_board_np # Store the validated board

        if not solver.is_solvable(self.initial_board_np):
            messagebox.showwarning("Unsolvable Puzzle", "This N-Puzzle configuration is not solvable based on inversion count.")
            self.result_text_area.insert(tk.END, "This puzzle configuration is NOT SOLVABLE.\n")
            self.result_text_area.config(state=tk.DISABLED)
            self._set_status("Puzzle is unsolvable.", "red")
            return
        
        algo_name_full = self.selected_algorithm.get()
        goal_state_np = solver.get_goal_state(self.current_n)

        self.result_text_area.insert(tk.END, f"Solving with {algo_name_full} for board:\n{str(self.initial_board_np)}\nGoal:\n{str(goal_state_np)}\n\nPlease wait...\n")
        self.master.update_idletasks()

        path, nodes_expanded, elapsed_time, iterations_ida = None, 0, 0.0, 0
        solution_found = False

        try:
            if algo_name_full == "BFS":
                path, nodes_expanded, elapsed_time = solver.bfs(self.initial_board_np, goal_state_np)
            elif algo_name_full == "DFS":
                path, nodes_expanded, elapsed_time = solver.dfs(self.initial_board_np, goal_state_np)
            elif "A*" in algo_name_full:
                path, nodes_expanded, elapsed_time = solver.a_star(self.initial_board_np, goal_state_np, solver.manhattan_distance)
            elif "IDA*" in algo_name_full:
                path, nodes_expanded, iterations_ida, elapsed_time = solver.ida_star(self.initial_board_np, goal_state_np, solver.manhattan_distance)
            
            solution_found = (path is not None)

        except Exception as e:
            self.result_text_area.insert(tk.END, f"An error occurred during solving: {e}\n")
            import traceback
            self.result_text_area.insert(tk.END, traceback.format_exc() + "\n")
            self._set_status(f"Error during {algo_name_full}.", "red")
        
        finally: # Ensure text area is disabled
            if solution_found:
                self.result_text_area.insert(tk.END, "\nSOLUTION FOUND!\n")
                self.result_text_area.insert(tk.END, "Move the blank tile (0) according to the path:\n")
                self.result_text_area.insert(tk.END, " -> ".join(path) + "\n\n")
                self.result_text_area.insert(tk.END, f"Number of steps: {len(path)}\n")
                self._set_status(f"Solved: {algo_name_full} in {len(path)} steps!", "green")
            else:
                # Only show this if no exception occurred but path is None
                if 'e' not in locals() or not e:
                    self.result_text_area.insert(tk.END, "\nNo solution found (or algorithm limit reached).\n")
                    self._set_status(f"No solution found by {algo_name_full}.", "orange")

            self.result_text_area.insert(tk.END, f"Nodes expanded: {nodes_expanded}\n")
            if "IDA*" in algo_name_full and iterations_ida > 0:
                self.result_text_area.insert(tk.END, f"IDA* iterations: {iterations_ida}\n")
            self.result_text_area.insert(tk.END, f"Time taken: {elapsed_time:.4f} seconds\n")
            self.result_text_area.config(state=tk.DISABLED)

    def _validate_and_handle_entry(self, var_name, index, mode, r, c):
        # Chỉ kiểm tra là số hoặc rỗng, không kiểm tra trùng lặp, không kiểm tra ngoài khoảng
        entry_widget = self.entry_widgets[r][c]
        current_value_str = entry_widget.get()
        if not current_value_str:  # Cell cleared
            setattr(entry_widget, 'last_valid_value', None)
            self._update_remaining_numbers_display()
            self._clear_status()
            return True

        if not current_value_str.isdigit():
            self._set_status(f"Cell ({r+1},{c+1}): Must be a number.", "red")
            entry_widget.delete(0, tk.END)
            return False

        num = int(current_value_str)
        setattr(entry_widget, 'last_valid_value', num)
        self._update_remaining_numbers_display()
        self._clear_status()
        return True

if __name__ == '__main__':
    root = tk.Tk()
    app = NPuzzleGUI(root)
    root.mainloop()