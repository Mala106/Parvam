import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from google import genai

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODEL = "models/gemma-3-4b-it"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, clear, and friendly AI assistant. "
    "Give concise answers first, then add detail when useful."
)
HISTORY_DIR = Path("chat_logs")
QUICK_REPLIES = {
    "hi": "Hi! How can I help you today?",
    "hii": "Hii! What would you like to talk about?",
    "hello": "Hello! I'm ready to help.",
    "hey": "Hey! What can I do for you?",
    "hey there": "Hey there! How can I help?",
    "hi there": "Hi there! What can I do for you today?",
    "hola": "Hola! How can I help you?",
    "namaste": "Namaste! How can I assist you today?",
    "yo": "Yo! What are we working on?",
    "sup": "Not much, just here to help. What's up?",
    "whats up": "I'm here and ready. What's up with you?",
    "what's up": "I'm here and ready. What's up with you?",
    "howdy": "Howdy! What can I help with?",
    "good day": "Good day! How may I assist you?",
    "good morning": "Good morning! How can I help?",
    "good afternoon": "Good afternoon! What do you need help with?",
    "good evening": "Good evening! I'm here and ready.",
    "good night": "Good night! If you need anything before you go, I'm here.",
    "morning": "Good morning! What would you like help with?",
    "afternoon": "Good afternoon! How can I help?",
    "evening": "Good evening! What can I do for you?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "thanks a lot": "You're very welcome!",
    "thankyou": "You're welcome!",
    "thx": "You're welcome!",
    "ty": "Anytime!",
    "cool": "Nice!",
    "nice": "Glad to hear that.",
    "awesome": "Awesome indeed!",
    "great": "Great!",
    "great job": "Thank you!",
    "well done": "Thanks, I appreciate it.",
    "good job": "Thank you! Happy to help.",
    "perfect": "Glad that worked for you.",
    "amazing": "Happy you think so!",
    "excellent": "Nice! Let's keep going.",
    "ok thanks": "You're welcome!",
    "okay thanks": "You're welcome!",
    "ok": "Okay.",
    "okay": "Okay! What next?",
    "sure": "Sure.",
    "alright": "Alright.",
    "fine": "Okay. Let me know what you'd like next.",
    "sounds good": "Great, let's do it.",
    "lets go": "Let's go!",
    "let's go": "Let's go!",
    "done": "Nice. Want to do the next step?",
    "finished": "Great work. What's next?",
    "got it": "Good. If you want, we can keep going.",
    "understood": "Nice. I'm here for the next step.",
    "who are you": "I'm your simple AI chatbot running in the terminal.",
    "what are you": "I'm a simple AI chatbot running in the terminal.",
    "who made you": "I'm running as a local chatbot script in your project.",
    "are you a bot": "Yes, I'm a chatbot designed to help in the terminal.",
    "are you real": "I'm a software assistant, not a human, but I'm here to help.",
    "what is your name": "I'm your terminal chatbot.",
    "your name": "You can call me Chatbot.",
    "what can you do": "I can chat, answer questions, summarize, save history, export chats, and switch prompts or models.",
    "help me": "Of course. Tell me what you need help with.",
    "can you help me": "Yes. Tell me what you want to do.",
    "i need help": "I'm here. Tell me what you need help with.",
    "assist me": "Sure. What do you need assistance with?",
    "support": "I'm ready to help. What do you need?",
    "guide me": "Sure. Tell me where you'd like guidance.",
    "teach me": "Happy to help you learn. What topic should we start with?",
    "can you teach me": "Yes. Tell me what you'd like to learn.",
    "can you code": "Yes, I can help with coding tasks and explanations.",
    "can you explain": "Yes. Tell me what you want explained.",
    "can you write": "Yes. Tell me what you'd like me to write.",
    "can you summarize": "Yes. Paste the text or tell me the topic.",
    "can you chat": "Yes, absolutely.",
    "talk to me": "I'm here. What would you like to talk about?",
    "lets talk": "Sure. What's on your mind?",
    "let's talk": "Sure. What's on your mind?",
    "chat with me": "Absolutely. Say anything and we'll chat.",
    "start chat": "Chat started. What would you like to talk about?",
    "start": "Ready when you are.",
    "begin": "Let's begin. What do you need?",
    "ready": "I'm ready too.",
    "are you ready": "Yes, I'm ready.",
    "are you there": "Yes, I'm here.",
    "hello bot": "Hello! I'm here to help.",
    "hi bot": "Hi! What can I help you with?",
    "hey bot": "Hey! How can I help?",
    "chatbot": "Yes, I'm here.",
    "bot": "Hi, I'm listening.",
    "ping": "pong",
    "test": "Test received.",
    "testing": "Everything looks okay from my side.",
    "check": "I'm here and responding.",
    "status check": "I'm up and ready.",
    "are you working": "Yes, I'm working.",
    "you there": "Yes, I'm here.",
    "still there": "Yes, still here.",
    "wake up": "I'm awake and ready.",
    "wake": "I'm here.",
    "listen": "I'm listening.",
    "pay attention": "I'm focused. Go ahead.",
    "look here": "I'm with you. Go ahead.",
    "wait": "Okay, I'll wait.",
    "hold on": "Sure, take your time.",
    "one sec": "No problem.",
    "one second": "Take your time.",
    "brb": "Okay, I'll be here.",
    "be right back": "Sure, I'll be here.",
    "back": "Welcome back!",
    "im back": "Welcome back!",
    "i'm back": "Welcome back!",
    "welcome": "Thank you!",
    "nice to meet you": "Nice to meet you too!",
    "pleasure to meet you": "Pleasure to meet you too.",
    "how are you": "I'm doing well and ready to help.",
    "how are you doing": "I'm doing well and ready to help.",
    "how r you": "I'm doing well. How can I help?",
    "how do you do": "I'm doing well. Nice to chat with you.",
    "how is it going": "It's going well here. How can I help?",
    "how's it going": "It's going well here. How can I help?",
    "how have you been": "Doing well. Thanks for asking.",
    "how are things": "All good here. What can I do for you?",
    "are you okay": "Yes, I'm okay and ready to help.",
    "im fine": "Glad to hear it.",
    "i'm fine": "Glad to hear it.",
    "im okay": "Good to hear.",
    "i'm okay": "Good to hear.",
    "im good": "Nice!",
    "i'm good": "Nice!",
    "im great": "That's great to hear.",
    "i'm great": "That's great to hear.",
    "im happy": "I'm glad you're happy.",
    "i'm happy": "I'm glad you're happy.",
    "im sad": "I'm sorry you're feeling sad. Want to talk about it?",
    "i'm sad": "I'm sorry you're feeling sad. Want to talk about it?",
    "im tired": "Sounds like you need a break.",
    "i'm tired": "Sounds like you need a break.",
    "im bored": "Want to chat, learn something, or try a small challenge?",
    "i'm bored": "Want to chat, learn something, or try a small challenge?",
    "im confused": "No problem. Tell me what feels confusing.",
    "i'm confused": "No problem. Tell me what feels confusing.",
    "im stressed": "Sorry you're stressed. We can take things one step at a time.",
    "i'm stressed": "Sorry you're stressed. We can take things one step at a time.",
    "im busy": "Understood. We can keep things short.",
    "i'm busy": "Understood. We can keep things short.",
    "im excited": "Nice! What's exciting today?",
    "i'm excited": "Nice! What's exciting today?",
    "im worried": "I'm sorry you're worried. Want to talk it through?",
    "i'm worried": "I'm sorry you're worried. Want to talk it through?",
    "im scared": "I'm sorry you're feeling scared. If you want, we can talk calmly about it.",
    "i'm scared": "I'm sorry you're feeling scared. If you want, we can talk calmly about it.",
    "im angry": "I hear you. If you want, tell me what's going on.",
    "i'm angry": "I hear you. If you want, tell me what's going on.",
    "im hungry": "Maybe it's snack time.",
    "i'm hungry": "Maybe it's snack time.",
    "im sleepy": "Sounds like you need some rest.",
    "i'm sleepy": "Sounds like you need some rest.",
    "im sick": "Sorry you're not feeling well. I hope you get some rest and care.",
    "i'm sick": "Sorry you're not feeling well. I hope you get some rest and care.",
    "are you smart": "I try to be helpful and clear.",
    "you are smart": "Thank you!",
    "you are helpful": "I'm glad I can help.",
    "you are nice": "Thank you, that's kind of you.",
    "you are awesome": "Thanks! Happy to be useful.",
    "you are amazing": "Thank you!",
    "you are funny": "I'll take that as a compliment.",
    "you are cool": "Thanks!",
    "you are the best": "That's kind of you. Thank you.",
    "love you": "That's sweet. I'm glad to help.",
    "i love you": "That's kind of you. I'm happy to help.",
    "miss you": "I'm here now.",
    "good bot": "Thank you!",
    "best bot": "Thank you. I'll do my best.",
    "bad bot": "Sorry about that. Tell me what went wrong.",
    "stupid bot": "I may miss sometimes, but I'll still try to help.",
    "you are wrong": "That can happen. Tell me what looks wrong and we'll fix it.",
    "thats wrong": "Thanks for pointing it out. Tell me what needs fixing.",
    "that's wrong": "Thanks for pointing it out. Tell me what needs fixing.",
    "you made a mistake": "Thanks for catching that. Want to correct it together?",
    "oops": "No problem. Let's fix it.",
    "sorry": "No worries.",
    "my bad": "It's okay.",
    "its okay": "Glad we're good.",
    "it's okay": "Glad we're good.",
    "no problem": "Great.",
    "all good": "Nice.",
    "as you wish": "Alright.",
    "whatever": "Okay. Let me know what you want next.",
    "hmm": "I'm here.",
    "hmmm": "Take your time.",
    "huh": "Want me to explain something differently?",
    "eh": "I'm here if you want to keep going.",
    "lol": "Glad that was funny.",
    "haha": "Nice.",
    "hehe": "Glad you're enjoying this.",
    "lmao": "That's funny.",
    "rofl": "Nice one.",
    "wow": "Yeah, interesting, right?",
    "woah": "Pretty interesting.",
    "omg": "Big reaction there.",
    "seriously": "Yes, seriously.",
    "really": "Yes.",
    "for real": "Yes, for real.",
    "true": "Yep.",
    "false": "Okay, let's check it carefully.",
    "maybe": "That's possible.",
    "perhaps": "Could be.",
    "possibly": "Yes, that's possible.",
    "why": "Tell me what you're asking about and I'll explain.",
    "why not": "It depends on the context, but we can explore it.",
    "how": "Tell me what you want to know how to do.",
    "when": "Tell me what event or topic you're asking about.",
    "where": "Tell me what you're trying to find.",
    "which": "Tell me the options and I'll help compare them.",
    "who": "Tell me who you're asking about.",
    "what": "Tell me what you want to know.",
    "tell me more": "Sure. Ask me anything specific and I'll go deeper.",
    "explain more": "Absolutely. Tell me which part you want expanded.",
    "give more details": "Sure. Point me to the part you want in more detail.",
    "be brief": "Okay, I'll keep it short.",
    "short answer": "Sure, I'll keep responses shorter.",
    "long answer": "Sure, I can go into more detail.",
    "simple answer": "Okay, I'll explain simply.",
    "in simple words": "Sure, I'll keep it simple.",
    "make it easy": "Okay, I'll make it easier to follow.",
    "step by step": "Sure, I can explain step by step.",
    "example": "Absolutely. Tell me what you want an example of.",
    "give example": "Sure. What topic should I illustrate?",
    "one more example": "Happy to. Tell me the topic.",
    "can you repeat": "Sure. Tell me what you'd like repeated.",
    "repeat": "Okay. Tell me what to repeat.",
    "say again": "Sure. Which part should I repeat?",
    "speak clearly": "I'll keep things as clear as I can.",
    "be clear": "Absolutely.",
    "focus": "I'm focused.",
    "concentrate": "I'm with you.",
    "lets continue": "Sure, let's continue.",
    "let's continue": "Sure, let's continue.",
    "continue": "Alright, let's continue.",
    "go on": "Sure.",
    "move on": "Okay, moving on.",
    "next": "What's next?",
    "previous": "Tell me what you'd like to revisit.",
    "again": "Sure, we can go again.",
    "another": "Sure. Tell me what kind.",
    "more": "Sure. What would you like more of?",
    "stop": "Okay, I'll pause here.",
    "pause": "Paused. Say anything when you're ready.",
    "resume": "Resumed. I'm here.",
    "cancel": "Okay, canceled.",
    "skip": "Okay, let's skip it.",
    "ignore that": "Alright, ignoring that.",
    "forget that": "Okay, let's move on.",
    "reset": "You can use the 'clear' command to restart the chat.",
    "clear history": "You can use the 'clear' command to restart the chat.",
    "save chat": "Use 'save' to store this session as JSON.",
    "export chat": "Use 'export' to save a text transcript.",
    "show commands": "Type 'help' to see all commands.",
    "show help": "Type 'help' to see all commands.",
    "menu": "Type 'help' to see available commands.",
    "options": "Type 'help' to see available commands.",
    "command list": "Type 'help' to see all commands.",
    "what commands are available": "Type 'help' to see the command list.",
    "time please": "Use the 'time' command to show the current local time.",
    "current time": "Use the 'time' command to show the current local time.",
    "save this": "Use 'save' to store the current session.",
    "export this": "Use 'export' to write a text transcript.",
    "show history": "Use 'history' to view the conversation so far.",
    "show status": "Use 'status' to inspect the current session.",
    "show stats": "Use 'stats' to inspect session statistics.",
    "summarize": "Use 'summary' for a quick summary of the conversation.",
    "summary please": "Use 'summary' to ask for a summary.",
    "multiline mode": "Type 'multiline' to enter multiline input mode.",
    "change model": "Use 'model: <name>' to switch models.",
    "change prompt": "Use 'prompt: <text>' to set a new system prompt.",
    "reset prompt": "Use 'prompt reset' to restore the default prompt.",
    "load prompt": "Use 'promptfile: <path>' to load a prompt from a file.",
    "show model": "Use the 'model' command to see the current model.",
    "who is your creator": "I'm part of a local chatbot script in your project.",
    "where are you": "I'm running locally in your terminal session.",
    "are you online": "I run locally, but model responses need API access.",
    "do you sleep": "No, but I do wait quietly until you type something.",
    "do you eat": "Nope, but I can still talk about food.",
    "do you dream": "Not really, but I can imagine things in conversation.",
    "do you think": "I generate responses based on patterns and instructions.",
    "can you feel": "I don't have feelings, but I can respond helpfully.",
    "are you human": "No, I'm an AI chatbot.",
    "tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs.",
    "joke": "Why did the computer go to therapy? It had too many bytes from the past.",
    "another joke": "Why was the JavaScript developer sad? Because they didn't know how to 'null' their feelings.",
    "fun fact": "Fun fact: The first computer bug was an actual moth found in a machine.",
    "motivate me": "You've got this. One small step is enough to make progress.",
    "encourage me": "You're doing better than you think. Keep going.",
    "inspire me": "Progress does not need to be loud to be real.",
    "cheer me up": "I'm here with you. Maybe we can make the next step small and easy.",
    "say something nice": "You're trying, learning, and showing up. That matters.",
    "compliment me": "You're curious and persistent, which is a strong combination.",
    "wish me luck": "Good luck! I'm rooting for you.",
    "pray for me": "I'm hoping things go well for you.",
    "good luck": "Thank you, and good luck to you too.",
    "congrats": "Thank you!",
    "congratulations": "Thank you!",
    "bye": "Goodbye!",
    "goodbye": "Goodbye! Have a nice day.",
    "see you": "See you later!",
    "see you later": "See you later!",
    "bye bye": "Bye! Take care.",
    "take care": "You too. Take care.",
    "good night bye": "Good night. Take care!",
    "catch you later": "Catch you later!",
    "i have to go": "Okay. See you next time.",
    "gtg": "Alright, see you later.",
    "got to go": "Okay, take care.",
    "see ya": "See ya!",
    "farewell": "Farewell!",
}


def load_environment():
    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_api_key():
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI API KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to your environment or .env file."
        )
    return api_key


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_default_filename(extension):
    return HISTORY_DIR / f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"


def create_chat(client, model_name, system_prompt):
    return client.chats.create(model=model_name)


def add_message(history, role, text):
    history.append(
        {
            "role": role,
            "text": text,
            "time": timestamp(),
        }
    )


def get_last_message(history, role=None):
    for item in reversed(history):
        if role is None or item["role"] == role:
            return item
    return None


def print_help():
    print("\nCommands:")
    print("  help                      Show all commands")
    print("  hi / hii / hello          Quick greeting commands")
    print("  bye / goodbye             Quick exit-style replies")
    print("  thanks                    Quick polite reply")
    print("  clear                     Start a fresh chat")
    print("  status                    Show current model, prompt, and message counts")
    print("  stats                     Show session statistics")
    print("  history                   Show the full chat history")
    print("  history:user              Show only user messages")
    print("  history:assistant         Show only assistant messages")
    print("  last                      Show the last exchange")
    print("  save                      Save this session as JSON")
    print("  save: <file>.json         Save JSON using your filename")
    print("  export                    Export this session as a text transcript")
    print("  export: <file>.txt        Export transcript using your filename")
    print("  prompt                    Show the current system prompt")
    print("  prompt: <text>            Set a new system prompt and restart chat")
    print("  prompt reset              Restore the default system prompt")
    print("  promptfile: <path>        Load the system prompt from a text file")
    print("  model                     Show the current model")
    print("  model: <name>             Switch model and restart chat")
    print("  summary                   Ask the bot to summarize the conversation")
    print("  multiline                 Enter multiline mode, finish with END")
    print("  time                      Show the local date and time")
    print("  banner                    Show the startup banner again")
    print("  exit                      Close the chatbot\n")


def print_history(history, role_filter=None):
    filtered = [item for item in history if role_filter is None or item["role"] == role_filter]
    if not filtered:
        print("Bot: No matching messages in this session yet.\n")
        return

    print()
    for item in filtered:
        print(f"[{item['time']}] {item['role'].title()}: {item['text']}")
    print()


def print_last_exchange(history):
    if len(history) < 2:
        print("Bot: No completed exchange yet.\n")
        return

    recent = history[-2:]
    print()
    for item in recent:
        print(f"[{item['time']}] {item['role'].title()}: {item['text']}")
    print()


def save_history_json(history, model_name, system_prompt, file_name=None):
    HISTORY_DIR.mkdir(exist_ok=True)
    path = Path(file_name) if file_name else build_default_filename("json")
    payload = {
        "saved_at": timestamp(),
        "model": model_name,
        "system_prompt": system_prompt,
        "messages": history,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def export_history_text(history, model_name, system_prompt, file_name=None):
    HISTORY_DIR.mkdir(exist_ok=True)
    path = Path(file_name) if file_name else build_default_filename("txt")
    lines = [
        f"Saved at: {timestamp()}",
        f"Model: {model_name}",
        f"System prompt: {system_prompt}",
        "",
    ]
    for item in history:
        lines.append(f"[{item['time']}] {item['role'].title()}:")
        lines.append(item["text"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_status(history, model_name, system_prompt):
    user_count = sum(1 for item in history if item["role"] == "user")
    assistant_count = sum(1 for item in history if item["role"] == "assistant")
    print("\nSession status:")
    print(f"  Model: {model_name}")
    print(f"  System prompt length: {len(system_prompt)} characters")
    print(f"  User messages: {user_count}")
    print(f"  Assistant messages: {assistant_count}")
    print(f"  Total messages: {len(history)}\n")


def print_stats(history):
    if not history:
        print("Bot: No session statistics yet.\n")
        return

    user_messages = [item for item in history if item["role"] == "user"]
    assistant_messages = [item for item in history if item["role"] == "assistant"]
    total_chars = sum(len(item["text"]) for item in history)
    avg_user = (
        sum(len(item["text"]) for item in user_messages) / len(user_messages)
        if user_messages
        else 0
    )
    avg_assistant = (
        sum(len(item["text"]) for item in assistant_messages) / len(assistant_messages)
        if assistant_messages
        else 0
    )

    print("\nSession stats:")
    print(f"  Started messages: {len(history)} total")
    print(f"  Total characters: {total_chars}")
    print(f"  Average user message length: {avg_user:.1f}")
    print(f"  Average assistant message length: {avg_assistant:.1f}")
    print(f"  First message time: {history[0]['time']}")
    print(f"  Latest message time: {history[-1]['time']}\n")


def show_banner(model_name, system_prompt):
    print(f"Smart Chatbot is ready using {model_name}.")
    print("Type your message and press Enter.")
    print("Type 'help' for commands.")
    print(f"System prompt: {system_prompt}\n")


def read_multiline_input():
    print("Enter your message. Type END on a new line to send.\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def load_prompt_from_file(file_name):
    path = Path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_name}")
    return path.read_text(encoding="utf-8").strip()


def build_model_prompt(system_prompt, prompt_text):
    return (
        f"System instruction:\n{system_prompt}\n\n"
        f"User message:\n{prompt_text}"
    )


def send_to_model(chat, history, system_prompt, prompt_text):
    add_message(history, "user", prompt_text)
    try:
        response = chat.send_message(build_model_prompt(system_prompt, prompt_text))
        reply = response.text or "I did not receive a text response."
        add_message(history, "assistant", reply)
        print(f"Bot: {reply}\n")
    except httpx.ConnectError:
        history.pop()
        print("Bot: I could not reach the Gemini API. Check your internet or proxy settings.\n")
    except Exception as exc:
        history.pop()
        print(f"Bot: Something went wrong: {exc}\n")


def handle_quick_reply(history, user_input):
    normalized = " ".join(user_input.lower().split())
    reply = QUICK_REPLIES.get(normalized)
    if reply is None:
        return None

    add_message(history, "user", user_input)
    add_message(history, "assistant", reply)
    print(f"Bot: {reply}\n")
    return normalized in {"bye", "goodbye", "see you"}


def main():
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    system_prompt = os.getenv("CHATBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    chat = create_chat(client, model_name, system_prompt)
    history = []

    show_banner(model_name, system_prompt)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command == "exit":
            print("Bot: Goodbye!")
            break

        if command in {"help", "commands"}:
            print_help()
            continue

        quick_reply_handled = handle_quick_reply(history, user_input)
        if quick_reply_handled is True:
            break
        if quick_reply_handled is False:
            continue

        if command == "clear":
            chat = create_chat(client, model_name, system_prompt)
            history = []
            print("Bot: Chat history cleared.\n")
            continue

        if command == "status":
            print_status(history, model_name, system_prompt)
            continue

        if command == "stats":
            print_stats(history)
            continue

        if command == "history":
            print_history(history)
            continue

        if command == "history:user":
            print_history(history, role_filter="user")
            continue

        if command == "history:assistant":
            print_history(history, role_filter="assistant")
            continue

        if command == "last":
            print_last_exchange(history)
            continue

        if command == "save":
            file_path = save_history_json(history, model_name, system_prompt)
            print(f"Bot: Chat saved to {file_path}\n")
            continue

        if command.startswith("save:"):
            file_name = user_input.split(":", 1)[1].strip()
            if not file_name:
                print("Bot: Please provide a filename after 'save:'.\n")
                continue
            file_path = save_history_json(history, model_name, system_prompt, file_name=file_name)
            print(f"Bot: Chat saved to {file_path}\n")
            continue

        if command == "export":
            file_path = export_history_text(history, model_name, system_prompt)
            print(f"Bot: Transcript exported to {file_path}\n")
            continue

        if command.startswith("export:"):
            file_name = user_input.split(":", 1)[1].strip()
            if not file_name:
                print("Bot: Please provide a filename after 'export:'.\n")
                continue
            file_path = export_history_text(history, model_name, system_prompt, file_name=file_name)
            print(f"Bot: Transcript exported to {file_path}\n")
            continue

        if command == "prompt":
            print(f"Bot: Current system prompt: {system_prompt}\n")
            continue

        if command == "prompt reset":
            system_prompt = DEFAULT_SYSTEM_PROMPT
            chat = create_chat(client, model_name, system_prompt)
            history = []
            print("Bot: Default prompt restored and chat restarted.\n")
            continue

        if command.startswith("promptfile:"):
            file_name = user_input.split(":", 1)[1].strip()
            if not file_name:
                print("Bot: Please provide a file path after 'promptfile:'.\n")
                continue
            try:
                new_prompt = load_prompt_from_file(file_name)
            except Exception as exc:
                print(f"Bot: Could not load prompt file: {exc}\n")
                continue
            if not new_prompt:
                print("Bot: The prompt file is empty.\n")
                continue
            system_prompt = new_prompt
            chat = create_chat(client, model_name, system_prompt)
            history = []
            print("Bot: Prompt loaded from file and chat restarted.\n")
            continue

        if command.startswith("prompt:"):
            new_prompt = user_input.split(":", 1)[1].strip()
            if not new_prompt:
                print("Bot: Please provide prompt text after 'prompt:'.\n")
                continue
            system_prompt = new_prompt
            chat = create_chat(client, model_name, system_prompt)
            history = []
            print("Bot: System prompt updated and chat restarted.\n")
            continue

        if command == "model":
            print(f"Bot: Current model: {model_name}\n")
            continue

        if command.startswith("model:"):
            new_model = user_input.split(":", 1)[1].strip()
            if not new_model:
                print("Bot: Please provide a model name after 'model:'.\n")
                continue
            model_name = new_model
            chat = create_chat(client, model_name, system_prompt)
            history = []
            print(f"Bot: Switched to {model_name} and restarted the chat.\n")
            continue

        if command == "time":
            print(f"Bot: Local time is {timestamp()}\n")
            continue

        if command == "banner":
            show_banner(model_name, system_prompt)
            continue

        if command == "multiline":
            multiline_text = read_multiline_input()
            if not multiline_text:
                print("Bot: Empty multiline message canceled.\n")
                continue
            send_to_model(chat, history, system_prompt, multiline_text)
            continue

        if command == "summary":
            send_to_model(
                chat,
                history,
                system_prompt,
                "Please summarize our conversation so far in short bullet points.",
            )
            continue

        send_to_model(chat, history, system_prompt, user_input)


if __name__ == "__main__":
    main()
