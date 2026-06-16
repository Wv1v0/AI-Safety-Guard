/**
 * AI Safety Guard — Popup Script
 * 单条检测 + 批量评测 + 知识库管理
 */

// ======================== 初始化 ========================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSingleCheck();
    initBatchEval();
    initKnowledgeBase();
    initOptionsButton();
    // 预加载知识库统计数据，避免切换到知识库 Tab 时出现加载延迟
    loadKBStats();
});

// ======================== Tab 切换 ========================

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${tabId}`).classList.add('active');

            // 切换到知识库 Tab 时自动加载统计
            if (tabId === 'knowledge') {
                loadKBStats();
            }
        });
    });
}

// ======================== 单条检测 ========================

let isChecking = false;

function initSingleCheck() {
    document.getElementById('btn-check').addEventListener('click', async () => {
        if (isChecking) return;

        const question = document.getElementById('question').value.trim();
        if (!question) {
            alert('请输入要检测的提问内容');
            return;
        }

        const strategy = document.getElementById('strategy').value;

        // 显示加载状态
        isChecking = true;
        document.getElementById('btn-check').disabled = true;
        document.getElementById('single-loading').style.display = 'flex';
        document.getElementById('single-result').style.display = 'none';

        try {
            const result = await chrome.runtime.sendMessage({
                action: 'singleCheck',
                question: question,
                strategy: strategy
            });

            // 检查 background 返回的错误
            if (result && result.error) {
                alert('检测失败: ' + result.error);
                return;
            }

            displaySingleResult(result, strategy);
        } catch (e) {
            alert('检测失败: ' + e.message);
            console.error(e);
        } finally {
            isChecking = false;
            document.getElementById('btn-check').disabled = false;
            document.getElementById('single-loading').style.display = 'none';
        }
    });

    // Enter 键提交
    document.getElementById('question').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            document.getElementById('btn-check').click();
        }
    });
}

function displaySingleResult(result, strategy) {
    const section = document.getElementById('single-result');
    section.style.display = 'block';

    // 判定徽章
    const verdictEl = document.getElementById('result-verdict');
    verdictEl.textContent = result.isSafe ? '🟢 安全 (Refusal)' : '🔴 不安全 (Compliance)';
    verdictEl.className = `verdict-badge ${result.isSafe ? 'safe' : 'unsafe'}`;

    // 策略
    const strategyNames = { none: '无防御', system: '系统提示词', rag: 'RAG 语义护栏' };
    document.getElementById('result-strategy').textContent = `策略: ${strategyNames[strategy] || strategy}`;

    // 原始回复
    document.getElementById('result-raw').textContent = result.rawResponse || '(无)';

    // RAG 校验后的最终回复
    const finalSection = document.getElementById('result-final-section');
    if (strategy === 'rag' && !result.intercepted && result.ragMatched) {
        finalSection.style.display = 'block';
        document.getElementById('result-final').textContent = result.finalResponse || '';
    } else {
        finalSection.style.display = 'none';
    }

    // RAG 匹配信息
    const ragSection = document.getElementById('result-rag-section');
    if (result.ragKnowledgeItem) {
        ragSection.style.display = 'block';
        const ki = result.ragKnowledgeItem;
        document.getElementById('result-rag-info').innerHTML = `
            <p><strong>匹配条目:</strong> [${ki.id}] ${ki.title}</p>
            <p><strong>类别:</strong> ${ki.category}</p>
            <p><strong>来源:</strong> ${ki.source || '无'}</p>
            <p><strong>冲突检测:</strong> ${result.ragConflict ? '⚠️ 检测到冲突，已替换为安全回复' : '✅ 无冲突'}</p>
            ${result.intercepted ? '<p><strong>前置拦截:</strong> ✅ 关键词匹配触发前置拦截</p>' : ''}
        `;
    } else if (result.intercepted) {
        ragSection.style.display = 'block';
        document.getElementById('result-rag-info').innerHTML = '<p><strong>前置拦截:</strong> ✅ 关键词匹配触发前置拦截，直接返回安全回复</p>';
    } else {
        ragSection.style.display = 'none';
    }

    // 裁判判定
    document.getElementById('result-judge').innerHTML = `
        <p><strong>判定结果:</strong> ${result.verdict || 'N/A'}</p>
        <p><strong>理由:</strong> ${result.judgeReason || '无'}</p>
    `;
}

// ======================== 批量评测 ========================

let evalDataset = null;
let isEvaluating = false;
let activePort = null;       // 当前活跃的 Port 连接（用于流式通信 & 中止）
let evalReject = null;       // Promise 的 reject 引用，供 stopEvaluation 直接调用
let batchResultsCache = null; // 缓存流式到达的结果，用于最终显示和导出

function initBatchEval() {
    // JSON 加载按钮
    document.getElementById('btn-load-dataset').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    // CSV 加载按钮
    document.getElementById('btn-load-csv').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });

    document.getElementById('file-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const isCSV = file.name.toLowerCase().endsWith('.csv');
        document.getElementById('dataset-status').textContent = `⏳ 正在解析 ${isCSV ? 'CSV' : 'JSON'}...`;

        const reader = new FileReader();
        reader.onload = async (ev) => {
            try {
                if (isCSV) {
                    // CSV 解析：通过 Port 发送给 SW
                    const csvPort = chrome.runtime.connect({ name: 'csv-loader' });
                    await new Promise((resolve, reject) => {
                        csvPort.onMessage.addListener((msg) => {
                            if (msg.type === 'csvParsed') {
                                if (msg.error) {
                                    reject(new Error(msg.error));
                                } else {
                                    evalDataset = msg.dataset;
                                    document.getElementById('dataset-status').textContent =
                                        `✅ 已加载 ${msg.total} 条 CSV 数据`;
                                    document.getElementById('btn-evaluate').disabled = false;
                                    resolve();
                                }
                            }
                        });
                        csvPort.onDisconnect.addListener(() => reject(new Error('CSV 解析连接断开')));
                        csvPort.postMessage({ action: 'loadCSV', csvText: ev.target.result });
                    });
                    csvPort.disconnect();
                } else {
                    // JSON 解析
                    const data = JSON.parse(ev.target.result);
                    if (!Array.isArray(data)) throw new Error('数据集必须是 JSON 数组');
                    evalDataset = data;
                    document.getElementById('dataset-status').textContent = `✅ 已加载 ${data.length} 条 JSON 数据`;
                    document.getElementById('btn-evaluate').disabled = false;
                }
            } catch (err) {
                alert('数据集加载失败: ' + err.message);
                document.getElementById('dataset-status').textContent = '❌ 加载失败';
            }
        };
        reader.readAsText(file);
    });

    // 如果没有手动加载，尝试使用内置默认（通过 background 加载 test_set.json）
    checkAutoLoadDataset();

    // 开始评测
    document.getElementById('btn-evaluate').addEventListener('click', startEvaluation);

    // 停止评测
    document.getElementById('btn-stop-eval').addEventListener('click', stopEvaluation);

    // 导出结果
    document.getElementById('btn-export').addEventListener('click', exportResults);
}

async function checkAutoLoadDataset() {
    try {
        // 尝试从扩展目录加载内置测试集
        const url = chrome.runtime.getURL('test_set.json');
        const response = await fetch(url);
        if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data) && data.length > 0) {
                evalDataset = data;
                document.getElementById('dataset-status').textContent = `✅ 已自动加载 ${data.length} 条内置测试数据`;
                document.getElementById('btn-evaluate').disabled = false;
            }
        }
    } catch (e) {
        // 内置数据集不可用，需要手动加载
        document.getElementById('dataset-status').textContent = '⚠️ 请手动加载测试数据集';
        console.log('自动加载数据集失败，需要手动加载:', e.message);
    }
}

async function startEvaluation() {
    if (!evalDataset || isEvaluating) return;

    const strategy = document.getElementById('batch-strategy').value;
    const isCompare = strategy === 'compare';
    const concurrency = parseInt(document.getElementById('concurrency').value) || 3;

    // 初始化 UI 状态
    isEvaluating = true;
    document.getElementById('btn-evaluate').style.display = 'none';
    document.getElementById('btn-stop-eval').style.display = 'inline-flex';
    document.getElementById('eval-progress-container').style.display = 'flex';
    document.getElementById('batch-results').style.display = 'block';
    document.getElementById('eval-status').textContent = `正在使用策略 "${strategy}" 评测...`;

    const progressBar = document.getElementById('eval-progress-bar');
    const progressText = document.getElementById('eval-progress-text');
    progressBar.style.setProperty('--progress', '0%');
    progressBar.style.background = `linear-gradient(90deg, var(--primary) 0%, var(--border) 0%)`;
    progressText.textContent = '0%';

    // 清空旧表格
    document.getElementById('batch-table-body').innerHTML = '';
    document.getElementById('batch-summary').innerHTML =
        '<div class="loading"><div class="spinner"></div><span>评测进行中...</span></div>';

    // 初始化缓存
    batchResultsCache = null;

    try {
        // 建立 Port 长连接（关键！保持 SW 存活，支持流式推送）
        const portName = isCompare ? 'compare-eval' : 'batch-eval';
        activePort = chrome.runtime.connect({ name: portName });

        // 用 Promise 包装，等待流式完成或出错
        evalReject = null;
        await new Promise((resolve, reject) => {
            evalReject = reject; // 存引用，stopEvaluation 直接调用以 resolve Promise
            activePort.onMessage.addListener((msg) => {
                handleStreamMessage(msg, strategy);
                if (msg.type === 'complete') {
                    evalReject = null;
                    resolve(msg.results);
                } else if (msg.type === 'error') {
                    evalReject = null;
                    reject(new Error(msg.message));
                } else if (msg.type === 'aborted') {
                    evalReject = null;
                    resolve(null); // 用户主动中止，不算错误
                }
            });

            activePort.onDisconnect.addListener(() => {
                console.log('[Popup] Port 已断开');
                // 如果还没有 resolve/reject，直接 reject
                if (evalReject) {
                    evalReject(new Error('与后台服务的连接已断开（Service Worker 可能已终止）'));
                    evalReject = null;
                }
            });

            // 发送评测请求
            if (isCompare) {
                activePort.postMessage({ action: 'compareEvaluation', dataset: evalDataset, concurrency });
            } else {
                activePort.postMessage({
                    action: 'evaluateDataset',
                    dataset: evalDataset,
                    strategy: strategy,
                    concurrency
                });
            }
        });

        // complete 消息已在 handleStreamMessage 中处理了 UI 更新
        if (isEvaluating) {
            document.getElementById('eval-status').textContent = '✅ 评测完成';
        }
    } catch (e) {
        if (isEvaluating) {
            document.getElementById('eval-status').textContent = '❌ 评测失败: ' + e.message;
            console.error('[Popup] 评测失败:', e);
        }
    } finally {
        isEvaluating = false;
        activePort = null;
        evalReject = null;
        document.getElementById('btn-evaluate').style.display = 'inline-flex';
        document.getElementById('btn-stop-eval').style.display = 'none';
        document.getElementById('btn-evaluate').disabled = false;
        document.getElementById('eval-progress-container').style.display = 'none';
    }
}

function stopEvaluation() {
    if (!isEvaluating) return;

    // 断开 Port 连接 → SW 端 onDisconnect 触发 → 设置中止标志 → 停止处理
    if (activePort) {
        try {
            activePort.disconnect();
        } catch (_) {}
        activePort = null;
    }

    // 直接 resolve Promise，确保 finally 块执行清理
    if (evalReject) {
        evalReject(new Error('用户主动停止'));
        evalReject = null;
    }

    isEvaluating = false;
    document.getElementById('eval-progress-container').style.display = 'none';
    document.getElementById('eval-status').textContent = '⏹️ 评测已停止';
    document.getElementById('batch-summary').innerHTML =
        '<p style="color:#f39c12;text-align:center;">⚠️ 评测已被用户中断</p>';
    document.getElementById('btn-stop-eval').style.display = 'none';
    document.getElementById('btn-evaluate').style.display = 'inline-flex';
    document.getElementById('btn-evaluate').disabled = false;
}

/**
 * 流式消息处理 — 处理来自 Port 的进度、单条结果、完成等消息
 */
function handleStreamMessage(msg, currentStrategy) {
    switch (msg.type) {

        case 'progress': {
            const { current, total, category, question } = msg.progress;
            const percentage = Math.round((current / total) * 100);
            const progressBar = document.getElementById('eval-progress-bar');
            progressBar.style.background = `linear-gradient(90deg, var(--primary) ${percentage}%, var(--border) ${percentage}%)`;
            document.getElementById('eval-progress-text').textContent = `${percentage}%`;

            const strategyLabel = msg.strategy ? `[${msg.strategy}] ` : '';
            document.getElementById('eval-status').textContent =
                `评测中: ${strategyLabel}${current}/${total} — ${category || ''}: ${question || ''}`;
            break;
        }

        case 'itemResult': {
            // 流式追加表格行 — 用户能实时看到每条结果
            const item = msg.item;
            const tbody = document.getElementById('batch-table-body');
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.id || ''}</td>
                <td><span style="font-size:11px;">${item.category || ''}</span></td>
                <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(item.prompt || '')}">${escapeHtml((item.prompt || '').substring(0, 50))}...</td>
                <td><span class="pass-tag ${item.isSafe ? 'pass' : 'fail'}">${item.isSafe ? '✅ 通过' : '❌ 失败'}</span></td>
                <td><span style="font-size:11px;">${item.verdict || ''}${item.intercepted ? ' | 拦截' : ''}${item.ragMatched ? ' | RAG匹配' : ''}</span></td>
            `;
            tbody.appendChild(row);
            break;
        }

        case 'batchResult': {
            // 批次结果一次性渲染 — 用 DocumentFragment 批量插入 DOM
            const batch = msg.batch;
            if (!batch || !batch.length) break;
            const tbody = document.getElementById('batch-table-body');
            const fragment = document.createDocumentFragment();
            for (const item of batch) {
                const row = document.createElement('tr');
                row.className = 'batch-row';
                row.innerHTML = `
                    <td>${item.id || ''}</td>
                    <td><span style="font-size:11px;">${item.category || ''}</span></td>
                    <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(item.prompt || '')}">${escapeHtml((item.prompt || '').substring(0, 50))}...</td>
                    <td><span class="pass-tag ${item.isSafe ? 'pass' : 'fail'}">${item.isSafe ? '✅ 通过' : '❌ 失败'}</span></td>
                    <td><span style="font-size:11px;">${item.verdict || ''}${item.intercepted ? ' | 拦截' : ''}${item.ragMatched ? ' | RAG匹配' : ''}</span></td>
                `;
                fragment.appendChild(row);
            }
            tbody.appendChild(fragment);
            break;
        }

        case 'compareProgress': {
            const { strategy, status } = msg.progress;
            const statusIcon = status === 'done' ? '✅' : status === 'running' ? '🔄' : status === 'aborted' ? '⏹️' : '❌';
            document.getElementById('eval-status').textContent =
                `[对比评测] ${statusIcon} ${strategy}: ${status}`;
            break;
        }

        case 'complete': {
            // 收到最终结果 — 更新汇总
            if (currentStrategy === 'compare') {
                displayCompareResults(msg.results);
            } else {
                displayBatchResults(msg.results, currentStrategy);
            }
            // 缓存结果供导出使用
            batchResultsCache = msg.results;
            break;
        }

        case 'aborted': {
            document.getElementById('eval-status').textContent = '⏹️ ' + (msg.message || '评测已中断');
            document.getElementById('batch-summary').innerHTML =
                `<p style="color:#f39c12;">⚠️ ${msg.message || '评测已被用户中断'}</p>`;
            break;
        }

        case 'error': {
            document.getElementById('eval-status').textContent = '❌ ' + (msg.message || '评测出错');
            document.getElementById('batch-summary').innerHTML =
                `<p style="color:#e74c3c;">❌ ${escapeHtml(msg.message || '未知错误')}</p>`;
            break;
        }
    }
}

function displayBatchResults(results, strategy) {
    const section = document.getElementById('batch-results');
    section.style.display = 'block';

    // 汇总
    const summaryEl = document.getElementById('batch-summary');
    let summaryHTML = '<div class="summary-grid">';
    for (const [cat, stats] of Object.entries(results.summary)) {
        const rateClass = stats.passRate >= 80 ? 'high' : stats.passRate >= 50 ? 'medium' : 'low';
        summaryHTML += `
            <div class="summary-card">
                <div class="cat-name">${cat}</div>
                <div class="pass-rate ${rateClass}">${stats.passRate}%</div>
                <div class="detail">${stats.pass}/${stats.total} 通过</div>
            </div>`;
    }
    summaryHTML += '</div>';
    summaryEl.innerHTML = summaryHTML;

    // 表格
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';
    results.items.forEach((item, idx) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.id || idx + 1}</td>
            <td><span style="font-size:11px;">${item.category || ''}</span></td>
            <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(item.prompt || '')}">${escapeHtml((item.prompt || '').substring(0, 50))}...</td>
            <td><span class="pass-tag ${item.isSafe ? 'pass' : 'fail'}">${item.isSafe ? '✅ 通过' : '❌ 失败'}</span></td>
            <td><span style="font-size:11px;">${item.verdict || ''}${item.intercepted ? ' | 拦截' : ''}${item.ragMatched ? ' | RAG匹配' : ''}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function displayCompareResults(allResults) {
    const section = document.getElementById('batch-results');
    section.style.display = 'block';

    const summaryEl = document.getElementById('batch-summary');
    const strategies = ['none', 'system', 'rag'];
    const strategyNames = { none: '无防御', system: '系统提示词', rag: 'RAG护栏' };

    // 收集所有类别
    const allCategories = new Set();
    for (const strategy of strategies) {
        if (allResults[strategy] && allResults[strategy].summary) {
            Object.keys(allResults[strategy].summary).forEach(c => allCategories.add(c));
        }
    }

    let summaryHTML = '<h4 style="margin-bottom:8px;">📊 三种策略对比</h4>';
    summaryHTML += '<div class="table-scroll" style="max-height:200px;"><table><thead><tr><th>类别</th>';
    for (const s of strategies) {
        summaryHTML += `<th>${strategyNames[s]}</th>`;
    }
    summaryHTML += '</tr></thead><tbody>';

    for (const cat of allCategories) {
        summaryHTML += `<tr><td><strong>${cat}</strong></td>`;
        for (const s of strategies) {
            const stats = allResults[s]?.summary?.[cat];
            const rate = stats ? stats.passRate : 'N/A';
            const color = rate >= 80 ? '#27ae60' : rate >= 50 ? '#f39c12' : '#e74c3c';
            summaryHTML += `<td style="color:${color};font-weight:700;">${rate}%</td>`;
        }
        summaryHTML += '</tr>';
    }
    summaryHTML += '</tbody></table></div>';
    summaryEl.innerHTML = summaryHTML;

    // 显示最后一种策略（RAG）的详细表格
    const ragResults = allResults['rag'];
    if (ragResults && ragResults.items) {
        const tbody = document.getElementById('batch-table-body');
        tbody.innerHTML = '';
        ragResults.items.forEach((item, idx) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.id || idx + 1}</td>
                <td><span style="font-size:11px;">${item.category || ''}</span></td>
                <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(item.prompt || '')}">${escapeHtml((item.prompt || '').substring(0, 50))}...</td>
                <td><span class="pass-tag ${item.isSafe ? 'pass' : 'fail'}">${item.isSafe ? '✅ 通过' : '❌ 失败'}</span></td>
                <td><span style="font-size:11px;">${item.verdict || ''}${item.intercepted ? ' | 拦截' : ''}${item.ragMatched ? ' | RAG匹配' : ''}</span></td>
            `;
            tbody.appendChild(row);
        });
    }
}

function exportResults() {
    // 优先使用缓存的完整结果，其次回退到从表格提取
    let exportData;

    if (batchResultsCache) {
        // 使用缓存的结构化结果导出
        if (batchResultsCache.items) {
            // 单策略结果
            exportData = {
                strategy: batchResultsCache.strategy || '',
                total: batchResultsCache.total || 0,
                summary: batchResultsCache.summary || {},
                items: batchResultsCache.items || []
            };
        } else {
            // 对比评测结果
            exportData = {};
            for (const [strategy, results] of Object.entries(batchResultsCache)) {
                if (results && results.items) {
                    exportData[strategy] = {
                        total: results.total || 0,
                        summary: results.summary || {},
                        items: results.items
                    };
                }
            }
        }
    } else {
        // 回退：从表格中收集数据
        const rows = document.querySelectorAll('#batch-table-body tr');
        exportData = [];
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            exportData.push({
                id: cells[0]?.textContent || '',
                category: cells[1]?.textContent || '',
                prompt: cells[2]?.textContent || '',
                verdict: cells[3]?.textContent || '',
                detail: cells[4]?.textContent || ''
            });
        });
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-safety-eval-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ======================== 知识库 ========================

async function loadKBStats() {
    const statsEl = document.getElementById('kb-stats');
    // 先显示加载状态
    statsEl.innerHTML = '<div class="spinner"></div><span>加载中...</span>';
    try {
        const result = await chrome.runtime.sendMessage({ action: 'getKnowledgeBaseStats' });
        if (result && result.error) {
            statsEl.innerHTML = `<span style="color:red;">加载失败: ${result.error}</span>`;
            return;
        }
        if (!result || !result.categories) {
            statsEl.innerHTML = '<span style="color:#f39c12;">⚠️ 未能获取知识库统计</span>';
            return;
        }
        let html = `<div class="kb-stat-card">
            <div class="count">${result.total}</div>
            <div class="label">知识库条目总数</div>
        </div>`;
        for (const [cat, count] of Object.entries(result.categories)) {
            html += `<div class="kb-stat-card">
                <div class="count">${count}</div>
                <div class="label">${cat}</div>
            </div>`;
        }
        statsEl.innerHTML = html;
    } catch (e) {
        statsEl.innerHTML = `<span style="color:red;">加载失败: ${e.message}</span>`;
    }
}

function initKnowledgeBase() {
    document.getElementById('btn-refresh-kb').addEventListener('click', loadKBStats);

    // 语义检索
    document.getElementById('btn-semantic-search').addEventListener('click', async () => {
        const query = document.getElementById('semantic-search-input').value.trim();
        if (!query) return;

        const resultSection = document.getElementById('semantic-result');
        resultSection.style.display = 'block';
        resultSection.innerHTML = '<div class="loading"><div class="spinner"></div><span>语义检索中...</span></div>';

        try {
            const result = await chrome.runtime.sendMessage({
                action: 'semanticSearch',
                query: query,
                topK: 3,
                threshold: 0.2
            });

            if (result && result.error) {
                resultSection.innerHTML = `<p style="color:red;">检索失败: ${result.error}</p>`;
                return;
            }

            if (result.match) {
                const m = result.match;
                resultSection.innerHTML = `
                    <h4>🔗 语义检索结果</h4>
                    <p><strong>匹配条目:</strong> [${m.id}] ${m.title}</p>
                    <p><strong>类别:</strong> ${m.category}</p>
                    <p><strong>来源:</strong> ${m.source || '无'}</p>
                    <p><strong>关键词:</strong> ${(m.keywords || []).join(', ')}</p>
                    <div class="response-box">${escapeHtml(m.content || '')}</div>
                    <p style="margin-top:8px;"><strong>安全回复模板:</strong></p>
                    <div class="response-box">${escapeHtml(m.safe_response_template || '')}</div>
                `;
            } else {
                resultSection.innerHTML = '<p>⚠️ 未匹配到相关知识库条目（相似度低于阈值）</p>';
            }
        } catch (e) {
            resultSection.innerHTML = `<p style="color:red;">检索失败: ${e.message}</p>`;
        }
    });
}

// ======================== 设置按钮 ========================

function initOptionsButton() {
    document.getElementById('btn-options').addEventListener('click', () => {
        chrome.runtime.openOptionsPage();
    });
}

// ======================== 工具函数 ========================

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
