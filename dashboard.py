import customtkinter as ctk
import paho.mqtt.client as mqtt
import os
import json
from datetime import datetime

# --- Global Config ---
BROKER = "localhost"
PORT = 1883
LOG_DIR = "mqtt_logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class GovernanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Zero-Trust Physical Key Governance")
        self.geometry("1100x750")
        
        # State Management
        # Keys 1-24: Key 2 is "out"
        self.keys = {f"Key {i}": ("in" if i != 2 else "out") for i in range(1, 25)}
        self.all_logs = []
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (LoginPage, MainDashboard):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        try:
            self.client.connect(BROKER, PORT, 60)
            self.client.subscribe("#")
            self.client.loop_start()
        except:
            pass

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        ts = datetime.now().strftime("%H:%M:%S")
        self.all_logs.append((ts, msg.topic, payload))
        if "LogsView" in self.frames["MainDashboard"].sub_frames:
             self.frames["MainDashboard"].sub_frames["LogsView"].refresh()

# --- LOGIN PAGE ---
class LoginPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white")
        self.controller = controller

        # Centered Login UI
        login_box = ctk.CTkFrame(self, width=600, height=350, corner_radius=40, 
                                 border_width=2, border_color="#1a4d66", fg_color="white")
        login_box.place(relx=0.5, rely=0.5, anchor="center")
        login_box.pack_propagate(False)

        ctk.CTkLabel(login_box, text="Username:", font=("Arial", 22), text_color="black").place(x=50, y=70)
        self.user_ent = ctk.CTkEntry(login_box, width=320, height=50, fg_color="#1a5d7a", corner_radius=10)
        self.user_ent.place(x=230, y=60)

        ctk.CTkLabel(login_box, text="Password:", font=("Arial", 22), text_color="black").place(x=50, y=140)
        self.pass_ent = ctk.CTkEntry(login_box, width=320, height=50, fg_color="#1a5d7a", corner_radius=10, show="*")
        self.pass_ent.place(x=230, y=130)

        login_btn = ctk.CTkButton(login_box, text="Login", font=("Arial", 22), 
                                 fg_color="white", text_color="black", border_width=2, 
                                 border_color="#1a4d66", width=320, height=50,
                                 command=lambda: controller.show_frame("MainDashboard"))
        login_btn.place(x=230, y=220)

# --- MAIN DASHBOARD ---
class MainDashboard(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="#f0f0f0")
        self.controller = controller
        
        # Sidebar Navigation
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, border_width=1, border_color="black")
        self.sidebar.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar, text="Control Centre", font=("Arial", 20, "bold")).pack(pady=20)
        
        self.nav_btns = {}
        items = ["Homepage", "Key Management", "Key Tagging", "ESP32-C3 Configuration", "Logs", "Logout"]
        for item in items:
            btn = ctk.CTkButton(self.sidebar, text=item, fg_color="white", text_color="black", 
                               corner_radius=10, border_width=1, border_color="black", height=45,
                               command=lambda i=item: self.switch_view(i))
            btn.pack(fill="x", padx=15, pady=8)
            self.nav_btns[item] = btn

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.sub_frames = {}
        for F in (HomeView, KeyMgmtView, BatchTaggingView, ESPConfigView, LogsView):
            self.sub_frames[F.name] = F(self.content, self.controller)
            self.sub_frames[F.name].place(relx=0, rely=0, relwidth=1, relheight=1)
            
        self.switch_view("Homepage")

    def switch_view(self, name):
        if name == "Logout": self.controller.show_frame("LoginPage")
        else:
            for n, b in self.nav_btns.items():
                b.configure(fg_color="#d9d9d9" if n == name else "white")
            self.sub_frames[name].tkraise()
            if hasattr(self.sub_frames[name], "refresh"): self.sub_frames[name].refresh()

# --- VIEWS ---

class HomeView(ctk.CTkFrame):
    name = "Homepage"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.place(relx=0.5, rely=0.5, anchor="center")
        # Dashboard Overview stats
        stats = [("10", "Total Keys"), ("2", "Key Issued"), ("8", "Key In Vault"),
                 ("4", "Total ESP32-C3"), ("3", "ESP32-C3 Online"), ("1", "ESP32-C3 Offline")]
        for i, (v, t) in enumerate(stats):
            box = ctk.CTkFrame(grid, width=180, height=130, border_width=1, border_color="black", corner_radius=15, fg_color="white")
            box.grid(row=i//3, column=i%3, padx=15, pady=15)
            box.pack_propagate(False)
            ctk.CTkLabel(box, text=v, font=("Arial", 28, "bold")).pack(pady=(30,0))
            ctk.CTkLabel(box, text=t, font=("Arial", 14)).pack()

class KeyMgmtView(ctk.CTkFrame):
    name = "Key Management"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Key Status", font=("Arial", 40, "bold")).pack(pady=10)
        
        self.scroll = ctk.CTkScrollableFrame(self, corner_radius=30, border_width=1, border_color="#1a4d66")
        self.scroll.pack(fill="both", expand=True, padx=30, pady=10)
        
        # Key Management Controls
        ctrl = ctk.CTkFrame(self, corner_radius=20, border_width=1, border_color="black", fg_color="white")
        ctrl.pack(fill="x", padx=30, pady=20)
        self.ent = ctk.CTkEntry(ctrl, placeholder_text="Key Number", width=250, height=45)
        self.ent.pack(side="left", padx=20, pady=15)
        ctk.CTkButton(ctrl, text="Add New Key", height=45, border_width=1, border_color="black", fg_color="white", text_color="black", command=self.add).pack(side="left", padx=10)
        ctk.CTkButton(ctrl, text="Remove Key", height=45, border_width=1, border_color="black", fg_color="white", text_color="black", command=self.rem).pack(side="left", padx=10)

    def refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        for i, (name, status) in enumerate(self.controller.keys.items()):
            color = "#7ed957" if status == "in" else "#ff3131" # Green if in vault, red if issued
            ctk.CTkButton(self.scroll, text=name, fg_color=color, text_color="black", border_width=1, 
                          width=100, height=45, hover_color=color).grid(row=i//6, column=i%6, padx=10, pady=10)

    def add(self):
        n = self.ent.get()
        if n: self.controller.keys[n] = "in"; self.refresh()
    def rem(self):
        n = self.ent.get()
        if n in self.controller.keys: del self.controller.keys[n]; self.refresh()

class BatchTaggingView(ctk.CTkFrame):
    name = "Key Tagging"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        self.selected_keys = set()
        
        ctk.CTkLabel(self, text="Batch Key Tagging", font=("Arial", 32, "bold")).pack(pady=10)
        
        self.scroll = ctk.CTkScrollableFrame(self, corner_radius=30, border_width=1, border_color="#1a4d66", fg_color="#d9d9d9")
        self.scroll.pack(fill="both", expand=True, padx=30, pady=10)

        # Batch Tagging Controls
        bottom = ctk.CTkFrame(self, corner_radius=20, border_width=1, border_color="black", fg_color="white")
        bottom.pack(fill="x", padx=30, pady=20)
        self.user = ctk.CTkEntry(bottom, placeholder_text="Enter Username", height=45, width=400)
        self.user.pack(side="left", padx=20, pady=15)
        
        ctk.CTkButton(bottom, text="Untag Key", fg_color="white", text_color="black", border_width=1, height=45, command=self.untag).pack(side="right", padx=10)
        ctk.CTkButton(bottom, text="Tag Key", fg_color="#3b8ed0", text_color="white", height=45, command=self.tag).pack(side="right", padx=10)

    def refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        for i, (name, status) in enumerate(self.controller.keys.items()):
            bg = "#3b8ed0" if name in self.selected_keys else "white" # Highlight selected keys in blue
            btn = ctk.CTkButton(self.scroll, text=name, fg_color=bg, text_color="black", border_width=1, 
                                width=100, height=45, command=lambda n=name: self.toggle_key(n))
            btn.grid(row=i//6, column=i%6, padx=10, pady=10)

    def toggle_key(self, name):
        if name in self.selected_keys: self.selected_keys.remove(name)
        else: self.selected_keys.add(name)
        self.refresh()

    def tag(self):
        print(f"Tagging {list(self.selected_keys)} to {self.user.get()}")
        self.selected_keys.clear(); self.refresh()

    def untag(self):
        print(f"Untagging {list(self.selected_keys)}")
        self.selected_keys.clear(); self.refresh()

class ESPConfigView(ctk.CTkFrame):
    name = "ESP32-C3 Configuration"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="ESP32-C3 Settings", font=("Arial", 40, "bold")).pack(pady=10)
        
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20)
        
        # Left Panel (Node Selection List)
        left = ctk.CTkFrame(main, border_width=1, border_color="black", corner_radius=30, fg_color="white")
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        node_scroll = ctk.CTkScrollableFrame(left, fg_color="transparent")
        node_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # 4 Nodes total for now
        for i in range(1, 5):
            node_id = f"ESP32-C3-00{i}"
            btn = ctk.CTkButton(node_scroll, text=f"ESP32 Node {i}", fg_color="white", text_color="black", border_width=1, 
                                command=lambda n=node_id: self.show_details(n))
            btn.pack(pady=10, padx=20, fill="x")
        
        # Right Panel (Node Specific Details)
        self.right = ctk.CTkFrame(main, border_width=1, border_color="black", corner_radius=30, fg_color="white")
        self.right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.detail_widgets = []
        self.placeholder = ctk.CTkLabel(self.right, text="Select a node to view details", text_color="gray")
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def show_details(self, node_id):
        self.placeholder.place_forget()
        for w in self.detail_widgets: w.destroy()
        self.detail_widgets = []

        # Hardcoded static and threshold values
        fields = [
            ("Node Number:", node_id),
            ("Connection:", "UP"),
            ("Door Status:", "CLOSE"),
            ("Door Open Threshold:", "100"),
            ("Door Close Threshold:", "100"),
            ("Human Detected Threshold:", "100")
        ]

        for lbl, val in fields:
            f = ctk.CTkFrame(self.right, fg_color="transparent")
            f.pack(fill="x", padx=30, pady=8)
            self.detail_widgets.append(f)
            ctk.CTkLabel(f, text=lbl, text_color="black", font=("Arial", 16)).pack(side="left")
            
            # Editable thresholds vs static labels
            if "Threshold" in lbl:
                e = ctk.CTkEntry(f, width=120); e.insert(0, val); e.pack(side="right")
            else:
                ctk.CTkLabel(f, text=val, text_color="gray").pack(side="right")

        btn_f = ctk.CTkFrame(self.right, fg_color="transparent")
        btn_f.pack(pady=20); self.detail_widgets.append(btn_f)
        ctk.CTkButton(btn_f, text="Cancel", fg_color="white", text_color="black", border_width=1, width=100).pack(side="left", padx=10)
        ctk.CTkButton(btn_f, text="Save", width=100, fg_color="#3b8ed0").pack(side="left", padx=10)

class LogsView(ctk.CTkFrame):
    name = "Logs"
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="white", border_width=1, border_color="black")
        self.controller = controller
        ctk.CTkLabel(self, text="Logs", font=("Arial", 40, "bold")).pack(pady=5)
        # Search Bar for Logs
        self.search = ctk.CTkEntry(self, placeholder_text="Search Bar", height=40)
        self.search.pack(fill="x", padx=30, pady=10)
        self.box = ctk.CTkTextbox(self, corner_radius=30, border_width=1, border_color="black")
        self.box.pack(fill="both", expand=True, padx=30, pady=10)

    def refresh(self):
        self.box.delete("1.0", "end")
        for ts, top, msg in self.controller.all_logs:
            self.box.insert("end", f"[{ts}] {top}: {msg}\n")

if __name__ == "__main__":
    app = GovernanceApp()
    app.mainloop()
