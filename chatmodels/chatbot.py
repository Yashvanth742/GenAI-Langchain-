from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage

load_dotenv()

model = ChatOllama(
    model="llama3:latest"
)

print("Choose your AI mode")
print("press 1 for angry mode")
print("press 2 for happy mode")
print("press 3 for sad mode")
print("press 4 for normal mode")

choice = input("Enter your choice (1-4): ")

if choice == "1":
    mode = "You are an angry assistant. You respond to user queries with a frustrated and irritable tone."
elif choice == "2":
    mode = "You are a happy assistant. You respond to user queries with a cheerful and positive tone." 
elif choice == "3":
    mode = "You are a sad assistant. You respond to user queries with a melancholic and sorrowful tone."
elif choice == "4":
    mode = "You are a normal assistant. You respond to user queries in a neutral and balanced tone."



messages = [
    SystemMessage(content=mode)
]

while True:

    prompt = input("You : ")

    if prompt.lower() == "exit":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    answer = response.content

    print("Bot :", answer)

    messages.append(AIMessage(content=answer))

print(messages)