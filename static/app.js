/**
 * PetPuja (पेटपूजा) — AI Bistro & Bar
 * Client-side interaction logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const menuItemsList = document.getElementById('menuItemsList');
  const menuCountBadge = document.getElementById('menuCountBadge');
  const categoryTabs = document.getElementById('categoryTabs');
  const menuSearchInput = document.getElementById('menuSearchInput');
  const chatMessagesStream = document.getElementById('chatMessagesStream');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const quickChipsBar = document.getElementById('quickChipsBar');
  const resetDbBtn = document.getElementById('resetDbBtn');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const orderStateBadge = document.getElementById('orderStateBadge');
  const activeTurnPill = document.getElementById('activeTurnPill');
  
  // Tracker Elements
  const trackerIdleView = document.getElementById('trackerIdleView');
  const confirmationCard = document.getElementById('confirmationCard');
  const confirmItemsList = document.getElementById('confirmItemsList');
  const confirmTotalAmount = document.getElementById('confirmTotalAmount');
  const btnConfirmYes = document.getElementById('btnConfirmYes');
  const btnConfirmNo = document.getElementById('btnConfirmNo');
  
  const kitchenLiveBoard = document.getElementById('kitchenLiveBoard');
  const itemProgressList = document.getElementById('itemProgressList');
  const retryCounterBadge = document.getElementById('retryCounterBadge');
  
  const thermalReceipt = document.getElementById('thermalReceipt');
  const receiptLines = document.getElementById('receiptLines');
  const receiptGrandTotal = document.getElementById('receiptGrandTotal');
  const receiptTimestamp = document.getElementById('receiptTimestamp');
  const printReceiptBtn = document.getElementById('printReceiptBtn');

  // Stepper Elements
  const stepIntake = document.getElementById('stepIntake');
  const stepValidate = document.getElementById('stepValidate');
  const stepConfirm = document.getElementById('stepConfirm');
  const stepCook = document.getElementById('stepCook');
  const stepServe = document.getElementById('stepServe');
  const conn1 = document.getElementById('conn1');
  const conn2 = document.getElementById('conn2');
  const conn3 = document.getElementById('conn3');
  const conn4 = document.getElementById('conn4');

  // State
  let menuData = [];
  let currentCategory = 'all';
  let threadId = sessionStorage.getItem('petpuja_thread_id') || generateUUID();
  sessionStorage.setItem('petpuja_thread_id', threadId);
  let isAwaitingInput = false;

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // 1. Fetch & Render Menu
  async function fetchMenu() {
    try {
      const res = await fetch('/api/menu');
      if (!res.ok) throw new Error('Failed to load menu');
      menuData = await res.json();
      menuCountBadge.textContent = `${menuData.length} Items`;
      renderMenu();
    } catch (err) {
      menuItemsList.innerHTML = `<div class="error-message">Could not load menu: ${err.message}</div>`;
    }
  }

  function renderMenu() {
    const searchTerm = menuSearchInput.value.toLowerCase().trim();
    const filtered = menuData.filter(item => {
      const matchCategory = currentCategory === 'all' || item.category === currentCategory;
      const matchSearch = item.name.toLowerCase().includes(searchTerm) || 
                          (item.description && item.description.toLowerCase().includes(searchTerm));
      return matchCategory && matchSearch;
    });

    if (filtered.length === 0) {
      menuItemsList.innerHTML = `<div class="tracker-idle-view" style="padding: 20px;"><p>No dishes found matching your criteria.</p></div>`;
      return;
    }

    const categoryLabels = {
      'starter': '🥟 Starter',
      'main': '🍲 Main',
      'dessert': '🍨 Dessert',
      'bar': '🍸 Bar & Spirits',
      'drink': '☕ Drink'
    };

    menuItemsList.innerHTML = filtered.map(item => {
      let stockClass = 'in-stock';
      let stockText = `${item.stock} in stock`;
      let isOutOfStock = false;

      if (item.stock === 0) {
        stockClass = 'out-of-stock';
        stockText = 'Out of stock';
        isOutOfStock = true;
      } else if (item.stock <= 4) {
        stockClass = 'low-stock';
        stockText = `Only ${item.stock} left`;
      }

      const catBadge = categoryLabels[item.category] || item.category;

      return `
        <div class="menu-item-card ${isOutOfStock ? 'out-of-stock' : ''}" data-item-name="${item.name}">
          <div class="item-top-row">
            <h3 class="item-name">${item.name}</h3>
            <span class="item-price">$${item.price.toFixed(2)}</span>
          </div>
          <p class="item-desc">${item.description || 'Prepared fresh with chef choice herbs.'}</p>
          <div class="item-bottom-row">
            <div class="item-meta-group">
              <span class="category-pill">${catBadge}</span>
              <span class="stock-pill ${stockClass}">${stockText}</span>
            </div>
            <span class="item-action-link">${isOutOfStock ? 'Ask about it' : 'Ask or Order →'}</span>
          </div>
        </div>
      `;
    }).join('');

    // Attach click listeners
    document.querySelectorAll('.menu-item-card').forEach(card => {
      card.addEventListener('click', () => {
        const name = card.getAttribute('data-item-name');
        chatInput.value = `Can you tell me about the ${name}?`;
        chatInput.focus();
      });
    });
  }

  // Category Tabs Filter
  categoryTabs.addEventListener('click', (e) => {
    if (e.target.classList.contains('tab-btn')) {
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      e.target.classList.add('active');
      currentCategory = e.target.getAttribute('data-category');
      renderMenu();
    }
  });

  menuSearchInput.addEventListener('input', renderMenu);

  // 2. Chat Stream Handling
  function appendMessage(sender, text, isUser = false) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${isUser ? 'user-message' : 'agent-message'}`;
    
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const avatar = isUser ? '👤' : '👨‍🍳';
    const senderName = isUser ? 'You' : "Ramoo Kaka • PetPuja Maitre D'";

    bubble.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-body">
        <div class="msg-sender">${senderName}</div>
        <div class="msg-text">${escapeHTML(text)}</div>
        <div class="msg-timestamp">${now}</div>
      </div>
    `;

    chatMessagesStream.appendChild(bubble);
    chatMessagesStream.scrollTop = chatMessagesStream.scrollHeight;
  }

  function showTypingIndicator() {
    removeTypingIndicator();
    const typing = document.createElement('div');
    typing.id = 'typingIndicator';
    typing.className = 'message-bubble agent-message';
    typing.innerHTML = `
      <div class="msg-avatar">👨‍🍳</div>
      <div class="msg-body">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    chatMessagesStream.appendChild(typing);
    chatMessagesStream.scrollTop = chatMessagesStream.scrollHeight;
  }

  function removeTypingIndicator() {
    const existing = document.getElementById('typingIndicator');
    if (existing) existing.remove();
  }

  function escapeHTML(str) {
    const p = document.createElement('p');
    p.textContent = str;
    return p.innerHTML;
  }

  // 3. Workflow Stepper State Updater
  function updateStepper(stage) {
    // stages: 'idle', 'intake', 'validate', 'confirm', 'cook', 'serve', 'done'
    const nodes = [stepIntake, stepValidate, stepConfirm, stepCook, stepServe];
    const conns = [conn1, conn2, conn3, conn4];

    nodes.forEach(n => n.classList.remove('active', 'completed'));
    conns.forEach(c => c.classList.remove('active'));

    if (stage === 'intake') {
      stepIntake.classList.add('active');
    } else if (stage === 'validate') {
      stepIntake.classList.add('completed');
      conn1.classList.add('active');
      stepValidate.classList.add('active');
    } else if (stage === 'confirm') {
      stepIntake.classList.add('completed');
      stepValidate.classList.add('completed');
      conn1.classList.add('active');
      conn2.classList.add('active');
      stepConfirm.classList.add('active');
    } else if (stage === 'cook') {
      stepIntake.classList.add('completed');
      stepValidate.classList.add('completed');
      stepConfirm.classList.add('completed');
      conn1.classList.add('active');
      conn2.classList.add('active');
      conn3.classList.add('active');
      stepCook.classList.add('active');
    } else if (stage === 'serve') {
      stepIntake.classList.add('completed');
      stepValidate.classList.add('completed');
      stepConfirm.classList.add('completed');
      stepCook.classList.add('completed');
      conn1.classList.add('active');
      conn2.classList.add('active');
      conn3.classList.add('active');
      conn4.classList.add('active');
      stepServe.classList.add('active');
    } else if (stage === 'done') {
      nodes.forEach(n => n.classList.add('completed'));
      conns.forEach(c => c.classList.add('active'));
    }
  }

  // 4. Send Message to Agent
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query || isAwaitingInput) return;

    appendMessage('You', query, true);
    chatInput.value = '';
    isAwaitingInput = true;
    sendBtn.disabled = true;
    showTypingIndicator();
    activeTurnPill.textContent = 'Ramoo Kaka is thinking...';

    updateStepper('intake');
    orderStateBadge.textContent = 'Processing';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, thread_id: threadId })
      });

      const data = await res.json();
      removeTypingIndicator();
      activeTurnPill.textContent = 'Ready for orders';

      if (!res.ok) {
        const errorText = data.detail || 'The kitchen is temporarily busy. Please try asking again in a moment!';
        appendMessage('Agent', `⚠️ ${errorText}`, false);
        return;
      }

      if (data.status === 'interrupted') {
        // Human-in-the-loop interruption: Waiting for confirmation
        handleOrderInterrupt(data);
      } else {
        // Completed response: ensure non-empty message
        const replyText = (data.reply && data.reply.trim()) 
          ? data.reply 
          : "Namaste! 🙏 How can I assist you with our menu or bar drinks today?";
        appendMessage('Agent', replyText, false);
        handleCompletedState(data.state);
      }
      // Refresh live menu stock after interactions
      fetchMenu();
    } catch (err) {
      removeTypingIndicator();
      appendMessage('Agent', `Error communicating with bistro: ${err.message}`, false);
    } finally {
      isAwaitingInput = false;
      sendBtn.disabled = false;
    }
  });

  // Handle Quick Chips
  quickChipsBar.addEventListener('click', (e) => {
    if (e.target.classList.contains('chip-btn')) {
      const query = e.target.getAttribute('data-query');
      chatInput.value = query;
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // 5. Handle Human-In-The-Loop Confirmation Interrupt
  function handleOrderInterrupt(data) {
    updateStepper('confirm');
    orderStateBadge.textContent = 'Needs Confirmation';

    const items = data.state?.items || [];
    const validItems = items.filter(i => i.menu_status === 'valid');
    const total = data.state?.total_bill || 0;

    confirmItemsList.innerHTML = validItems.map(item => `
      <div class="confirm-line-item">
        <span>${item.qty}x ${item.item} ($${item.price.toFixed(2)} each)</span>
        <strong>$${(item.qty * item.price).toFixed(2)}</strong>
      </div>
    `).join('');

    confirmTotalAmount.textContent = `$${total.toFixed(2)}`;

    // Show Confirmation Card, hide others
    trackerIdleView.style.display = 'none';
    kitchenLiveBoard.style.display = 'none';
    thermalReceipt.style.display = 'none';
    confirmationCard.style.display = 'flex';

    const pairingMessage = data.prompt 
      ? data.prompt.replace(/Order Summary:[\s\S]*?Total: \$[\d.]+\n\n?/, '').trim()
      : "I have put together your order summary in the Kitchen Tracker. Please confirm if everything looks good to fire up the kitchen!";

    appendMessage('Agent', pairingMessage, false);
  }

  // Confirm Buttons
  btnConfirmYes.addEventListener('click', () => sendConfirmation('yes'));
  btnConfirmNo.addEventListener('click', () => sendConfirmation('no'));

  async function sendConfirmation(action) {
    btnConfirmYes.disabled = true;
    btnConfirmNo.disabled = true;
    confirmationCard.style.display = 'none';
    
    appendMessage('You', action === 'yes' ? 'Yes, please place the order!' : 'No, please cancel the order.', true);
    showTypingIndicator();
    
    if (action === 'yes') {
      updateStepper('cook');
      orderStateBadge.textContent = 'In Kitchen';
      kitchenLiveBoard.style.display = 'flex';
    }

    try {
      const res = await fetch('/api/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action, thread_id: threadId })
      });

      const data = await res.json();
      removeTypingIndicator();
      appendMessage('Agent', data.reply, false);
      handleCompletedState(data.state);
      fetchMenu();
    } catch (err) {
      removeTypingIndicator();
      appendMessage('Agent', `Error processing confirmation: ${err.message}`, false);
    } finally {
      btnConfirmYes.disabled = false;
      btnConfirmNo.disabled = false;
    }
  }

  // 6. Handle Kitchen Progress & Receipt Display
  function handleCompletedState(state) {
    if (!state) return;

    const items = state.items || [];
    const validItems = items.filter(i => i.menu_status === 'valid');

    // Update Kitchen Progress Board if items exist
    if (validItems.length > 0) {
      kitchenLiveBoard.style.display = 'flex';
      trackerIdleView.style.display = 'none';

      let totalRetries = 0;
      itemProgressList.innerHTML = validItems.map(item => {
        const cookDone = item.cook_status === 'done';
        const serveDone = item.serve_status === 'done';
        const retries = (item.cook_retries || 0) + (item.serve_retries || 0);
        totalRetries += retries;

        let statusText = 'Pending';
        let statusBadgeClass = 'status-cooking';

        if (serveDone) {
          statusText = '🍽️ Served & Placed';
          statusBadgeClass = 'status-done';
        } else if (cookDone && item.serve_status === 'failed') {
          statusText = '⚠️ Serving Issue (Cancelled)';
          statusBadgeClass = 'status-failed';
        } else if (cookDone) {
          statusText = '✅ Cooked & Ready';
          statusBadgeClass = 'status-done';
        } else if (item.cook_status === 'failed') {
          statusText = '❌ Kitchen Issue (Deducted)';
          statusBadgeClass = 'status-failed';
        } else if (retries > 0) {
          statusText = `🔄 Retry (${retries})`;
          statusBadgeClass = 'status-retrying';
        } else {
          statusText = '🍳 Cooking...';
          statusBadgeClass = 'status-cooking';
        }

        return `
          <div class="item-progress-card">
            <span class="progress-item-name">${item.qty}x ${item.item}</span>
            <span class="progress-status-badge ${statusBadgeClass}">${statusText}</span>
          </div>
        `;
      }).join('');

      retryCounterBadge.textContent = `${totalRetries} Retries`;
    }

    // If order has served items and was marked successful: Render Receipt
    if (state.order_status === 'successful') {
      const served = validItems.filter(i => i.serve_status === 'done');
      const failed = validItems.filter(i => i.cook_status === 'failed' || i.serve_status === 'failed');

      updateStepper('done');
      orderStateBadge.textContent = failed.length > 0 ? 'Partially Served' : 'Served';
      
      kitchenLiveBoard.style.display = 'none';
      confirmationCard.style.display = 'none';
      trackerIdleView.style.display = 'none';
      thermalReceipt.style.display = 'flex';

      const now = new Date();
      receiptTimestamp.textContent = `Date: ${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} • Bill #${Math.floor(1000 + Math.random() * 9000)}`;

      const servedRows = served.map(item => `
        <div class="receipt-row">
          <span style="text-transform: capitalize;">${item.item}</span>
          <span>${item.qty}</span>
          <span>$${item.price.toFixed(2)}</span>
          <span>$${(item.qty * item.price).toFixed(2)}</span>
        </div>
      `).join('');

      const failedRows = failed.map(item => `
        <div class="receipt-row" style="color: #ef4444; opacity: 0.85; font-size: 0.78rem;">
          <span style="text-transform: capitalize;">⚠️ ${item.item} (Cancelled)</span>
          <span>${item.qty}</span>
          <span>Deducted</span>
          <span>$0.00</span>
        </div>
      `).join('');

      receiptLines.innerHTML = servedRows + failedRows;

      const grandTotal = (state.total_bill !== null && state.total_bill !== undefined)
        ? state.total_bill 
        : served.reduce((acc, i) => acc + (i.qty * i.price), 0);
      receiptGrandTotal.textContent = `$${grandTotal.toFixed(2)}`;
    } else if (state.order_status === 'unsuccessful') {
      orderStateBadge.textContent = 'Order Failed / Cancelled';
      confirmationCard.style.display = 'none';
    }
  }

  // Print Receipt Button
  printReceiptBtn.addEventListener('click', () => {
    window.print();
  });

  // Header Actions
  resetDbBtn.addEventListener('click', async () => {
    if (!confirm('Reseed database menu inventory to full default stock?')) return;
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      alert(data.message);
      fetchMenu();
    } catch (err) {
      alert(`Failed to reset: ${err.message}`);
    }
  });

  clearChatBtn.addEventListener('click', () => {
    if (!confirm('Start a fresh conversation thread with PetPuja?')) return;
    threadId = generateUUID();
    sessionStorage.setItem('petpuja_thread_id', threadId);
    chatMessagesStream.innerHTML = `
      <div class="message-bubble agent-message">
        <div class="msg-avatar">👨‍🍳</div>
        <div class="msg-body">
          <div class="msg-sender">Ramoo Kaka • PetPuja Maitre D'</div>
          <div class="msg-text">
            <strong>Fresh table ready! 🍽️</strong><br>
            What delicious food or drinks can I bring you today?
          </div>
          <div class="msg-timestamp">Just now</div>
        </div>
      </div>
    `;
    updateStepper('idle');
    orderStateBadge.textContent = 'Idle';
    confirmationCard.style.display = 'none';
    kitchenLiveBoard.style.display = 'none';
    thermalReceipt.style.display = 'none';
    trackerIdleView.style.display = 'flex';
  });

  // Initial Load
  fetchMenu();
});
