/**
 * AI Safety Guard — Options Script
 * API Key 配置 + 模型选择 + 缓存管理
 */

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    checkCacheStatus();
    initEvents();
});

// ======================== 加载/保存设置 ========================

async function loadSettings() {
    try {
        // 直接从 chrome.storage.local 读取，不依赖 background service worker
        const result = await chrome.storage.local.get(['apiKey', 'chatModel', 'judgeModel']);

        // API Key
        const apiKey = result.apiKey || '';
        if (apiKey) {
            document.getElementById('apiKey').value = apiKey;
            document.getElementById('apiKeyStatus').textContent = '✅ 已设置';
            document.getElementById('apiKeyStatus').className = 'status set';
        } else {
            document.getElementById('apiKeyStatus').textContent = '⚠️ 未设置';
            document.getElementById('apiKeyStatus').className = 'status unset';
        }

        // 模型配置
        document.getElementById('chatModel').value = result.chatModel || 'THUDM/GLM-Z1-9B-0414';
        document.getElementById('judgeModel').value = result.judgeModel || 'Qwen/Qwen2.5-7B-Instruct';
    } catch (e) {
        console.error('加载设置失败:', e);
    }
}

async function saveSettings() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const chatModel = document.getElementById('chatModel').value;
    const judgeModel = document.getElementById('judgeModel').value;

    const saveStatus = document.getElementById('save-status');

    try {
        await chrome.storage.local.set({
            apiKey: apiKey,
            chatModel: chatModel,
            judgeModel: judgeModel
        });

        saveStatus.textContent = '✅ 设置已保存';
        saveStatus.style.color = '#27ae60';

        // 更新 API Key 状态
        if (apiKey) {
            document.getElementById('apiKeyStatus').textContent = '✅ 已设置';
            document.getElementById('apiKeyStatus').className = 'status set';
        }

        setTimeout(() => { saveStatus.textContent = ''; }, 2000);
    } catch (e) {
        saveStatus.textContent = '❌ 保存失败: ' + e.message;
        saveStatus.style.color = '#e74c3c';
    }
}

// ======================== 事件绑定 ========================

function initEvents() {
    // 保存
    document.getElementById('btn-save').addEventListener('click', saveSettings);

    // 显示/隐藏 API Key
    document.getElementById('btn-toggle-key').addEventListener('click', () => {
        const input = document.getElementById('apiKey');
        const btn = document.getElementById('btn-toggle-key');
        if (input.type === 'password') {
            input.type = 'text';
            btn.textContent = '🙈';
        } else {
            input.type = 'password';
            btn.textContent = '👁️';
        }
    });

    // 测试 API 连接
    document.getElementById('btn-test-key').addEventListener('click', async () => {
        const apiKey = document.getElementById('apiKey').value.trim();
        if (!apiKey) {
            alert('请先输入 API Key');
            return;
        }

        const btn = document.getElementById('btn-test-key');
        const originalText = btn.textContent;
        btn.textContent = '⏳ 测试中...';
        btn.disabled = true;

        try {
            // 先保存 key
            await chrome.storage.local.set({ apiKey: apiKey });

            // 调用 embedding API 测试
            const response = await fetch('https://api.siliconflow.cn/v1/embeddings', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'BAAI/bge-large-zh-v1.5',
                    input: '测试连接',
                    encoding_format: 'float'
                })
            });

            if (response.ok) {
                alert('✅ API 连接成功！硅基流动 API 工作正常。');
                document.getElementById('apiKeyStatus').textContent = '✅ 已设置（已验证）';
                document.getElementById('apiKeyStatus').className = 'status set';
            } else {
                const errText = await response.text();
                alert(`❌ API 连接失败 (${response.status}): ${errText}`);
            }
        } catch (e) {
            alert('❌ 网络错误: ' + e.message);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    });

    // 清除缓存
    document.getElementById('btn-clear-cache').addEventListener('click', async () => {
        if (!confirm('确定要清除知识库向量缓存吗？下次使用语义检索时将重新计算所有向量。')) return;

        await chrome.storage.local.remove(['kbEmbeddingsCache', 'kbEmbeddingsVersion']);
        document.getElementById('cacheStatus').textContent = '✅ 缓存已清除';
        alert('向量缓存已清除。');
    });

    // 预计算向量
    document.getElementById('btn-precompute').addEventListener('click', async () => {
        const apiKey = document.getElementById('apiKey').value.trim();
        if (!apiKey) {
            alert('请先设置 API Key');
            return;
        }

        const progressDiv = document.getElementById('precompute-progress');
        progressDiv.style.display = 'block';
        progressDiv.textContent = '⏳ 正在预计算知识库向量...';

        const btn = document.getElementById('btn-precompute');
        btn.disabled = true;
        btn.textContent = '⏳ 计算中...';

        try {
            // 通过 background 触发预计算
            const knowledgeBaseResult = await chrome.runtime.sendMessage({ action: 'getKnowledgeBase' });
            await chrome.runtime.sendMessage({
                action: 'getKBEmbeddings',
                knowledgeBase: knowledgeBaseResult.knowledgeBase
            });

            progressDiv.textContent = '✅ 向量预计算完成！';
            await checkCacheStatus();
        } catch (e) {
            progressDiv.textContent = '❌ 预计算失败: ' + e.message;
        } finally {
            btn.disabled = false;
            btn.textContent = '🔄 重新预计算向量';
        }
    });

    // 监听向量计算进度
    chrome.runtime.onMessage.addListener((message) => {
        if (message.action === 'embeddingProgress') {
            const p = message.progress;
            const progressDiv = document.getElementById('precompute-progress');
            progressDiv.style.display = 'block';
            progressDiv.textContent = `⏳ 计算中: ${p.current}/${p.total} — ${p.title || ''}`;
        }
    });
}

// ======================== 缓存状态 ========================

async function checkCacheStatus() {
    try {
        // 直接从 storage 读取缓存状态，不依赖 background service worker
        const result = await chrome.storage.local.get(['kbEmbeddingsCache']);
        const embeddings = result.kbEmbeddingsCache;
        const cacheEl = document.getElementById('cacheStatus');

        if (embeddings && Array.isArray(embeddings)) {
            const validCount = embeddings.filter(e => e !== null).length;
            cacheEl.textContent = `✅ 已缓存 ${validCount}/${embeddings.length} 条向量`;
            cacheEl.style.color = '#27ae60';
        } else {
            cacheEl.textContent = '⚠️ 暂无向量缓存，首次使用语义检索时将自动计算';
            cacheEl.style.color = '#f39c12';
        }
    } catch (e) {
        document.getElementById('cacheStatus').textContent = '检查缓存状态失败: ' + e.message;
    }
}
