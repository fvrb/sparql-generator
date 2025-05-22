import tkinter as tk
from tkinter import ttk, scrolledtext
from app import run_pipeline, load_data

embeddings, data = load_data()

root = tk.Tk()
root.title("NLQ to SPARQL Generator")
root.geometry("400x600")
root.configure(padx=20, pady=20)

style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11))
style.configure("TCheckbutton", font=("Segoe UI", 11))
style.configure("TEntry", font=("Segoe UI", 11))

ttk.Label(root, text="Enter Natural Language Question:").grid(row=0, column=0, sticky="w")
nlq_entry = scrolledtext.ScrolledText(root, height=4, wrap=tk.WORD, font=("Segoe UI", 11))
nlq_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))

example_frame = ttk.Frame(root)
example_frame.grid(row=2, column=0, columnspan=2, sticky="w")

ttk.Label(example_frame, text="Number of similar examples supplied:").pack(side="left", padx=0)
example_count_var = tk.IntVar(value=3)
example_count_spinbox = ttk.Spinbox(example_frame, from_=1, to=10, textvariable=example_count_var, width=5)
example_count_spinbox.pack(side="left", padx=5)

cot_var = tk.BooleanVar()
cot_checkbox = ttk.Checkbutton(root, text="Enable Chain of Thought", variable=cot_var)
cot_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 15))

ttk.Label(root, text="Generated SPARQL Query:").grid(row=4, column=0, sticky="w")
sparql_output = scrolledtext.ScrolledText(root, height=12, wrap=tk.WORD, font=("Courier New", 11))
sparql_output.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
sparql_output.configure(state='disabled')

root.grid_rowconfigure(5, weight=1)
root.grid_columnconfigure(1, weight=1)


def on_generate():
    nlq = nlq_entry.get("1.0", tk.END).strip()
    n = example_count_var.get()
    cot = cot_var.get()

    sparql_output.configure(state='normal')
    sparql_output.delete("1.0", tk.END)

    if not nlq:
        sparql_output.insert(tk.END, "Please enter a natural language question.")
        sparql_output.configure(state='disabled')
        return

    sparql_output.insert(tk.END, "Generating SPARQL query...\n\n")
    sparql_output.update_idletasks()

    try:
        result = run_pipeline(nlq, embeddings, data, n, cot)
        sparql_output.delete("1.0", tk.END)
        sparql_output.insert(tk.END, result)
    except Exception as e:
        sparql_output.delete("1.0", tk.END)
        sparql_output.insert(tk.END, f"Error: {str(e)}")

    sparql_output.configure(state='disabled')


clipboard_icon = tk.PhotoImage(file="copy-icon.png").subsample(100, 100)


def copy_to_clipboard():
    root.clipboard_clear()
    sparql_output.configure(state='normal')
    text = sparql_output.get("1.0", tk.END).strip()
    sparql_output.configure(state='disabled')
    root.clipboard_append(text)

copy_button = ttk.Button(root, image=clipboard_icon, command=copy_to_clipboard)
copy_button.image = clipboard_icon
copy_button.grid(row=6, column=0, sticky="w")

generate_button = ttk.Button(root, text="Generate SPARQL", command=on_generate)
generate_button.grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()
