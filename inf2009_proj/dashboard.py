@@ class MainDashboard(ctk.CTkFrame):
         else:
             self.switch_view("Homepage")
+
+        # kick off periodic redraw so status dots update even if user is idle
+        self.after(2000, self._periodic_refresh)
+
+    def _periodic_refresh(self):
+        # refresh all frames that support it; this will update the node status
+        for frame in self.sub_frames.values():
+            if hasattr(frame, "refresh"):
+                try:
+                    frame.refresh()
+                except Exception:
+                    pass
+        # schedule again
+        self.after(2000, self._periodic_refresh)
@@
     def on_calibration_complete(self, node_id, state):
@@
         if node_id in self.controller.esp_nodes:
             self.controller.esp_nodes[node_id][state] = True
             self.controller._save_calib_states()
+
+            # if this was the last missing state, kick off training automatically
+            if all(self.controller.esp_nodes[node_id].values()):
+                if not self.controller.model_state.get(node_id, False):
+                    ts = datetime.now().strftime('%H:%M:%S')
+                    self.controller.all_logs.append((ts, "Model", f"Auto-training triggered for {node_id}"))
+                    # popup notification (non-blocking)
+                    popup = ctk.CTkToplevel(self)
+                    popup.title("Training Started")
+                    popup.geometry("360x120")
+                    ctk.CTkLabel(popup, text=f"Training model for {node_id}...", font=("Arial", 14)).pack(pady=20)
+                    self.after(3000, popup.destroy)
+                    # start training in foreground (blocks UI); we could thread later
+                    self.train_model(node_id)
@@
         # either we had no expectation, or the labels match
         if node_id in self.controller.esp_nodes:
             self.controller.esp_nodes[node_id][state] = True
             self.controller._save_calib_states()
@@
     def start_calibration(self, node_id):
         selected_state = self.radio_var.get()
         # remember choice so UI can restore it later
         self.controller.last_calib_choice[node_id] = selected_state
         # any fresh calibration invalidates previously trained model
         self.controller._mark_model_dirty(node_id)
         # record what we expect back
         self.controller.expected_calib[node_id] = selected_state
+
+        # wipe old CSI data for this node so subsequent training uses only
+        # the new recordings (per user request).
+        data_base = self.controller.csi_data_dir if os.path.exists(self.controller.csi_data_dir) else (
+            CSI_DATA_DIR if os.path.exists(CSI_DATA_DIR) else "csi_data")
+        node_dir = os.path.join(data_base, node_id)
+        if os.path.isdir(node_dir):
+            try:
+                shutil.rmtree(node_dir)
+                ts = datetime.now().strftime('%H:%M:%S')
+                self.controller.all_logs.append((ts, "Model", f"Cleared old CSI data for {node_id}"))
+            except Exception as e:
+                ts = datetime.now().strftime('%H:%M:%S')
+                self.controller.all_logs.append((ts, "Model", f"Failed to clear CSI data for {node_id}: {e}"))
@@
                 elif evt == "upload":
                     # upload progress from ESP32 (same handling as server-side "progress")
@@
                 elif evt == "state_change":
                     state = payload_json.get("state")
                     if isinstance(state, str):
                         self.node_detected_state[node_id] = state
                         self.all_logs.append((ts, "State", f"{node_id} -> {state}"))
+                elif evt == "model_ready":
+                    self.all_logs.append((ts, "Model", f"{node_id} reported model_ready"))
+                elif evt == "model_download_failed":
+                    self.all_logs.append((ts, "Model", f"{node_id} model download failed"))
@@
         # UI Refresh (thread-safe)
         def refresh_ui():
             if "MainDashboard" in self.frames:
                 subs = self.frames["MainDashboard"].sub_frames
@@
                 if completion_node and completion_state and "ESP32-C3 Configuration" in subs:
                     subs["ESP32-C3 Configuration"].on_calibration_complete(completion_node, completion_state)
                     # also log success/mismatch, the method itself handles the entry
+            # also refresh node status list in case detected state changed
+            if "MainDashboard" in self.frames:
+                cfg = self.frames["MainDashboard"].sub_frames.get("ESP32-C3 Configuration")
+                if cfg:
+                    cfg._refresh_node_status_list()
