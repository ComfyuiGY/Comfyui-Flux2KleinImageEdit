import { app } from "../../../scripts/app.js";

function updateDynamicInputs(node, targetCount) {
    console.log(`[Flux2Klein] updateDynamicInputs, targetCount=${targetCount}`);
    
    if (!node || !node.inputs) return;
    
    // 保存现有连接
    const connections = {};
    for (const input of node.inputs) {
        if (input.link !== null && input.name && input.name.startsWith("image_") && input.name !== "image_1") {
            const link = node.graph.links[input.link];
            if (link) {
                connections[input.name] = {
                    link: input.link,
                    targetId: link.target_id,
                    targetSlot: link.target_slot
                };
            }
        }
    }
    
    // 移除多余的输入（大于 targetCount 的）
    const toRemove = [];
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        if (input.name && input.name.startsWith("image_") && input.name !== "image_1") {
            const idx = parseInt(input.name.split("_")[1]);
            if (idx > targetCount) {
                toRemove.push(i);
                console.log(`[Flux2Klein] Removing ${input.name}`);
            }
        }
    }
    for (const idx of toRemove) {
        node.removeInput(idx);
    }
    
    // 添加缺失的输入（从 2 到 targetCount）
    for (let i = 2; i <= targetCount; i++) {
        const inputName = `image_${i}`;
        const exists = node.inputs.some(inp => inp.name === inputName);
        if (!exists) {
            console.log(`[Flux2Klein] Adding ${inputName}`);
            node.addInput(inputName, "IMAGE");
            // 恢复连接
            if (connections[inputName]) {
                const conn = connections[inputName];
                setTimeout(() => {
                    const targetSlot = node.inputs.findIndex(inp => inp.name === inputName);
                    if (targetSlot !== -1) {
                        node.graph.links[conn.link] = {
                            link_id: conn.link,
                            target_id: node.id,
                            target_slot: targetSlot,
                            source_id: conn.targetId,
                            source_slot: conn.targetSlot,
                            type: "IMAGE"
                        };
                        node.graph.afterChange();
                        if (app.canvas) app.canvas.setDirty(true);
                    }
                }, 50);
            }
        }
    }
    
    // 触发重绘
    node.setSize?.(node.computeSize());
    if (app.canvas) app.canvas.setDirty(true);
}

app.registerExtension({
    name: "ComfyUI.Flux2KleinImageEdit",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[Flux2Klein] beforeRegisterNodeDef", nodeData.name);
        
        if (nodeData.name !== "Flux2KleinImageEdit") {
            return;
        }
        
        console.log("[Flux2Klein] Registering for Flux2KleinImageEdit");
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            console.log("[Flux2Klein] onNodeCreated");
            const self = this;
            
            const inputcountWidget = this.widgets?.find(w => w.name === "inputcount");
            if (inputcountWidget) {
                setTimeout(() => {
                    updateDynamicInputs(self, inputcountWidget.value);
                }, 100);
                
                const originalCallback = inputcountWidget.callback;
                inputcountWidget.callback = function(value) {
                    console.log(`[Flux2Klein] inputcount changed to ${value}`);
                    if (originalCallback) originalCallback.call(this, value);
                    updateDynamicInputs(self, value);
                };
            }
            return result;
        };
        
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            const self = this;
            setTimeout(() => {
                const inputcountWidget = this.widgets?.find(w => w.name === "inputcount");
                if (inputcountWidget) {
                    updateDynamicInputs(self, inputcountWidget.value);
                }
            }, 200);
            return result;
        };
    }
});