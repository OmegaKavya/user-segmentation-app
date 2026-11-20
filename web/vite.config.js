import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Everything runs client side, so there is nothing to proxy.
export default defineConfig({ plugins: [react()], worker: { format: "es" } });
