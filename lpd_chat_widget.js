(function() {
    'use strict';
    
    // Chat Widget Class
    class LPDChatWidget {
        constructor() {
            this.isOpen = false;
            this.init();
        }
        
        init() {
            this.injectStyles();
            this.createChatButton();
            this.createChatInterface();
            this.bindEvents();
        }
        
        injectStyles() {
            const styles = `
                .lpd-chat-widget * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                .lpd-chat-button {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #1e7cff 0%, #0066ff 100%);
                    border-radius: 50%;
                    border: none;
                    cursor: pointer;
                    box-shadow: 0 4px 20px rgba(30, 124, 255, 0.4);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    color: white;
                    z-index: 10000;
                    transition: all 0.3s ease;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                }
                
                .lpd-chat-button:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 25px rgba(30, 124, 255, 0.6);
                }
                
                .lpd-chat-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.3);
                    z-index: 9998;
                    opacity: 0;
                    visibility: hidden;
                    transition: all 0.3s ease;
                }
                
                .lpd-chat-overlay.active {
                    opacity: 1;
                    visibility: visible;
                }
                
                .lpd-chat-container {
                    position: fixed;
                    bottom: 90px;
                    right: 20px;
                    width: 400px;
                    height: 600px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 10px 50px rgba(0, 0, 0, 0.3);
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    z-index: 9999;
                    transform: scale(0.7) translateY(50px);
                    opacity: 0;
                    visibility: hidden;
                    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                }
                
                .lpd-chat-container.active {
                    transform: scale(1) translateY(0);
                    opacity: 1;
                    visibility: visible;
                }
                
                @media (max-width: 480px) {
                    .lpd-chat-container {
                        width: calc(100vw - 20px);
                        height: calc(100vh - 40px);
                        bottom: 20px;
                        right: 10px;
                        left: 10px;
                        border-radius: 15px;
                    }
                }
                
                .lpd-chat-header {
                    background: linear-gradient(135deg, #1e7cff 0%, #0066ff 100%);
                    padding: 15px 20px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    color: white;
                }
                
                .lpd-header-left {
                    display: flex;
                    align-items: center;
                }
                
                .lpd-avatar {
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    background: white;
                    margin-right: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    color: #1e7cff;
                    font-size: 18px;
                }
                
                .lpd-assistant-name {
                    font-size: 16px;
                    font-weight: 600;
                }
                
                .lpd-close-btn {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                    padding: 5px 10px;
                    border-radius: 5px;
                    transition: background 0.2s;
                }
                
                .lpd-close-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                }
                
                .lpd-chat-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f5f5f5;
                }
                
                .lpd-message {
                    margin-bottom: 15px;
                }
                
                .lpd-message.bot {
                    display: flex;
                    justify-content: flex-start;
                }
                
                .lpd-message.user {
                    display: flex;
                    justify-content: flex-end;
                }
                
                .lpd-message-bubble {
                    max-width: 80%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    font-size: 14px;
                    line-height: 1.4;
                    word-wrap: break-word;
                }
                
                .lpd-message.bot .lpd-message-bubble {
                    background: #e5e5e7;
                    color: #333;
                    border-bottom-left-radius: 5px;
                }
                
                .lpd-message.user .lpd-message-bubble {
                    background: #007AFF;
                    color: white;
                    border-bottom-right-radius: 5px;
                }
                
                .lpd-message-actions {
                    display: flex;
                    gap: 10px;
                    margin-top: 8px;
                    padding-left: 5px;
                }
                
                .lpd-action-btn {
                    background: none;
                    border: none;
                    font-size: 18px;
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 50%;
                    transition: background 0.2s;
                }
                
                .lpd-action-btn:hover {
                    background: rgba(0, 0, 0, 0.1);
                }
                
                .lpd-thumbs-up {
                    color: #007AFF;
                }
                
                .lpd-thumbs-down {
                    color: #999;
                }
                
                .lpd-emergency-notice {
                    background: #f0f0f0;
                    padding: 12px 20px;
                    text-align: center;
                    font-size: 12px;
                    color: #666;
                    border-top: 1px solid #e0e0e0;
                }
                
                .lpd-input-container {
                    padding: 15px 20px;
                    background: white;
                    border-top: 1px solid #e0e0e0;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .lpd-input-wrapper {
                    flex: 1;
                    position: relative;
                }
                
                .lpd-message-input {
                    width: 100%;
                    padding: 12px 50px 12px 15px;
                    border: 1px solid #ddd;
                    border-radius: 25px;
                    font-size: 14px;
                    outline: none;
                    background: #f9f9f9;
                    font-family: inherit;
                }
                
                .lpd-message-input:focus {
                    border-color: #007AFF;
                    background: white;
                }
                
                .lpd-input-actions {
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    display: flex;
                    gap: 5px;
                }
                
                .lpd-input-action {
                    background: none;
                    border: none;
                    color: #999;
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 50%;
                    font-size: 16px;
                }
                
                .lpd-send-btn {
                    background: #007AFF;
                    border: none;
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    transition: background 0.2s;
                }
                
                .lpd-send-btn:hover {
                    background: #0056cc;
                }
                
                .lpd-send-btn:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                }
                
                .lpd-chat-messages::-webkit-scrollbar {
                    width: 4px;
                }
                
                .lpd-chat-messages::-webkit-scrollbar-track {
                    background: transparent;
                }
                
                .lpd-chat-messages::-webkit-scrollbar-thumb {
                    background: #ccc;
                    border-radius: 2px;
                }
                
                .lpd-typing-indicator {
                    display: none;
                    padding: 10px 0;
                }
                
                .lpd-typing-indicator.active {
                    display: block;
                }
                
                .lpd-typing-bubble {
                    background: #e5e5e7;
                    color: #999;
                    padding: 12px 16px;
                    border-radius: 18px;
                    border-bottom-left-radius: 5px;
                    font-size: 14px;
                    max-width: 80%;
                }
                
                .lpd-typing-dots {
                    display: inline-flex;
                    gap: 3px;
                }
                
                .lpd-typing-dot {
                    width: 6px;
                    height: 6px;
                    background: #999;
                    border-radius: 50%;
                    animation: lpd-typing 1.4s infinite ease-in-out;
                }
                
                .lpd-typing-dot:nth-child(1) { animation-delay: -0.32s; }
                .lpd-typing-dot:nth-child(2) { animation-delay: -0.16s; }
                
                @keyframes lpd-typing {
                    0%, 80%, 100% { opacity: 0.3; }
                    40% { opacity: 1; }
                }
            `;
            
            const styleSheet = document.createElement('style');
            styleSheet.textContent = styles;
            document.head.appendChild(styleSheet);
        }
        
        createChatButton() {
            this.chatButton = document.createElement('button');
            this.chatButton.className = 'lpd-chat-button';
            this.chatButton.innerHTML = '💬';
            this.chatButton.title = 'Chat with LPD Virtual Assistant';
            document.body.appendChild(this.chatButton);
        }
        
        createChatInterface() {
            // Create overlay
            this.overlay = document.createElement('div');
            this.overlay.className = 'lpd-chat-overlay';
            
            // Create chat container
            this.chatContainer = document.createElement('div');
            this.chatContainer.className = 'lpd-chat-container';
            
            this.chatContainer.innerHTML = `
                <div class="lpd-chat-header">
                    <div class="lpd-header-left">
                        <div class="lpd-avatar">👮</div>
                        <div class="lpd-assistant-name">LPD Virtual Assistant</div>
                    </div>
                    <button class="lpd-close-btn">Close</button>
                </div>

                <div class="lpd-chat-messages">
                    <div class="lpd-message bot">
                        <div class="lpd-message-bubble">
                            Keep in mind that at any time you can schedule time to speak with an officer using the scheduler icon at the bottom right. But it's a good idea to fill out the report before doing that.
                        </div>
                    </div>

                    <div class="lpd-message bot">
                        <div class="lpd-message-bubble">
                            OK, good, let's get started. Can you tell me where your bike was taken from?
                        </div>
                    </div>

                    <div class="lpd-message user">
                        <div class="lpd-message-bubble">
                            From my front lawn
                        </div>
                    </div>

                    <div class="lpd-message bot">
                        <div class="lpd-message-bubble">
                            Can you give me your address please?
                        </div>
                    </div>

                    <div class="lpd-message user">
                        <div class="lpd-message-bubble">
                            123 Perfect Way, BeautyTown, IL
                        </div>
                    </div>
                    
                    <div class="lpd-typing-indicator">
                        <div class="lpd-typing-bubble">
                            <div class="lpd-typing-dots">
                                <div class="lpd-typing-dot"></div>
                                <div class="lpd-typing-dot"></div>
                                <div class="lpd-typing-dot"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="lpd-emergency-notice">
                    If this is an emergency or you are in danger, please call 9-1-1.
                </div>

                <div class="lpd-input-container">
                    <div class="lpd-input-wrapper">
                        <input type="text" class="lpd-message-input" placeholder="Write a reply...">
                    </div>
                    <button class="lpd-send-btn">➤</button>
                </div>
            `;
            
            document.body.appendChild(this.overlay);
            document.body.appendChild(this.chatContainer);
        }
        
        bindEvents() {
            // Chat button click
            this.chatButton.addEventListener('click', () => this.toggleChat());
            
            // Overlay click to close
            this.overlay.addEventListener('click', () => this.closeChat());
            
            // Close button
            this.chatContainer.querySelector('.lpd-close-btn').addEventListener('click', () => this.closeChat());
            
            // Input and send functionality
            const messageInput = this.chatContainer.querySelector('.lpd-message-input');
            const sendBtn = this.chatContainer.querySelector('.lpd-send-btn');
            const chatMessages = this.chatContainer.querySelector('.lpd-chat-messages');
            
            sendBtn.addEventListener('click', () => this.sendMessage());
            
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
            
            messageInput.addEventListener('input', () => {
                const sendBtn = this.chatContainer.querySelector('.lpd-send-btn');
                sendBtn.disabled = messageInput.value.trim() === '';
            });
            
            // Action buttons - removed
            
            // Prevent chat from closing when clicking inside
            this.chatContainer.addEventListener('click', (e) => {
                e.stopPropagation();
            });
            
            // Initialize send button state
            sendBtn.disabled = true;
            
            // Escape key to close
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.isOpen) {
                    this.closeChat();
                }
            });
        }
        
        toggleChat() {
            if (this.isOpen) {
                this.closeChat();
            } else {
                this.openChat();
            }
        }
        
        openChat() {
            this.isOpen = true;
            this.overlay.classList.add('active');
            this.chatContainer.classList.add('active');
            this.chatButton.innerHTML = '✕';
            
            // Focus input after animation
            setTimeout(() => {
                const messageInput = this.chatContainer.querySelector('.lpd-message-input');
                messageInput.focus();
            }, 300);
        }
        
        closeChat() {
            this.isOpen = false;
            this.overlay.classList.remove('active');
            this.chatContainer.classList.remove('active');
            this.chatButton.innerHTML = '💬';
        }
        
        addMessage(text, isUser = false) {
            const chatMessages = this.chatContainer.querySelector('.lpd-chat-messages');
            const typingIndicator = chatMessages.querySelector('.lpd-typing-indicator');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `lpd-message ${isUser ? 'user' : 'bot'}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'lpd-message-bubble';
            bubble.textContent = text;
            
            messageDiv.appendChild(bubble);
            
            if (!isUser) {
                const actions = document.createElement('div');
                actions.className = 'lpd-message-actions';
                messageDiv.appendChild(actions);
            }
            
            chatMessages.insertBefore(messageDiv, typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        showTypingIndicator() {
            const typingIndicator = this.chatContainer.querySelector('.lpd-typing-indicator');
            const chatMessages = this.chatContainer.querySelector('.lpd-chat-messages');
            typingIndicator.classList.add('active');
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        hideTypingIndicator() {
            const typingIndicator = this.chatContainer.querySelector('.lpd-typing-indicator');
            typingIndicator.classList.remove('active');
        }
        
        sendMessage() {
            const messageInput = this.chatContainer.querySelector('.lpd-message-input');
            const text = messageInput.value.trim();
            
            if (text) {
                this.addMessage(text, true);
                messageInput.value = '';
                
                const sendBtn = this.chatContainer.querySelector('.lpd-send-btn');
                sendBtn.disabled = true;
                
                // Show typing indicator
                this.showTypingIndicator();
                
                // Simulate bot response after a delay
                setTimeout(() => {
                    this.hideTypingIndicator();
                    const responses = [
                        "Thank you for that information. Can you provide more details about what happened?",
                        "I understand. Let me help you with the next steps in filing your report.",
                        "That's helpful information. What else can you tell me about the incident?",
                        "I've recorded that information. Is there anything else you'd like to add to your report?",
                        "Got it. Can you describe any witnesses or additional details about the theft?",
                        "Thank you. I'll need a few more details to complete your police report."
                    ];
                    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
                    this.addMessage(randomResponse);
                }, 1500 + Math.random() * 1000);
            }
        }
    }
    
    // Initialize the widget when the DOM is ready
    function initializeLPDChat() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                new LPDChatWidget();
            });
        } else {
            new LPDChatWidget();
        }
    }
    
    // Auto-initialize
    initializeLPDChat();
    
    // Expose to global scope if needed
    window.LPDChatWidget = LPDChatWidget;
    
})();