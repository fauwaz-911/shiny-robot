// Cyber Execs Chatbot Widget — v2.0
// API URL auto-detects environment. Override by setting window.CHATBOT_API_URL before loading this script.

const API_URL = window.CHATBOT_API_URL || (
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/chat"
    : "/chat"   // same-origin when deployed on Render
);

const SESSION_ID = "session-" + Math.random().toString(36).substr(2, 9);

const chatButton   = document.getElementById("chat-button");
const chatPopup    = document.getElementById("chat-popup");
const chatBox      = document.getElementById("chat-box");
const sendBtn      = document.getElementById("send-btn");
const userInput    = document.getElementById("user-input");

// Open / close widget
chatButton.addEventListener("click", () => {
  chatPopup.classList.toggle("hidden");
  if (!chatPopup.classList.contains("hidden")) {
    userInput.focus();
    if (chatBox.children.length === 0) {
      addMessage("👋 Hi! I'm your Cyber Execs AI Assistant. Ask me anything about cybersecurity.", "bot");
    }
  }
});

// Send on button click or Enter
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function addMessage(text, sender) {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message-wrapper", sender === "user" ? "user-wrapper" : "bot-wrapper");

  const bubble = document.createElement("div");
  bubble.classList.add(sender === "user" ? "user-message" : "bot-message");
  bubble.innerText = text;

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping() {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message-wrapper", "bot-wrapper");
  wrapper.id = "typing-indicator";

  const bubble = document.createElement("div");
  bubble.classList.add("bot-message", "typing");
  bubble.innerHTML = "<span></span><span></span><span></span>";

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
  const indicator = document.getElementById("typing-indicator");
  if (indicator) indicator.remove();
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  sendBtn.disabled = true;
  showTyping();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        client_id: window.CHATBOT_CLIENT_ID || "default",
        session_id: SESSION_ID,
      }),
    });

    hideTyping();

    if (!response.ok) {
      addMessage("Something went wrong. Please try again.", "bot");
      return;
    }

    const data = await response.json();
    addMessage(data.reply || "No response received.", "bot");

  } catch (error) {
    hideTyping();
    addMessage("⚠️ Unable to reach the server. Please check your connection.", "bot");
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}
