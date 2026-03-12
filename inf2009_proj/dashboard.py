@@
         for node_id in selected_nodes:
             self.controller.last_calib_choice[node_id] = selected_state
             self.controller._mark_model_dirty(node_id)
             self.controller.expected_calib[node_id] = selected_state
+            # clear old CSI data for each node before starting calibration
+            data_base = self.controller.csi_data_dir if os.path.exists(self.controller.csi_data_dir) else (
+                CSI_DATA_DIR if os.path.exists(CSI_DATA_DIR) else "csi_data")
+            node_dir = os.path.join(data_base, node_id)
+            if os.path.isdir(node_dir):
+                try:
+                    shutil.rmtree(node_dir)
+                    ts = datetime.now().strftime('%H:%M:%S')
+                    self.controller.all_logs.append((ts, "Model", f"Cleared old CSI data for {node_id}"))
+                except Exception as e:
+                    ts = datetime.now().strftime('%H:%M:%S')
+                    self.controller.all_logs.append((ts, "Model", f"Failed to clear CSI data for {node_id}: {e}"))
             if self.controller._publish_collect_with_retry(node_id, selected_state, retries=3):
                 success_count += 1
             else:
                 failed_nodes.append(node_id)
