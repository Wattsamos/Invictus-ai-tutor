from Model_loader import load_model
from Prompt import build_prompt

try:
    tokenizer, model = load_model()
except Exception as error:
    print(f"Error loading model: {error}")


# Our foirst inference i.e to generate out put

def answer_question(question):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=100)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return(answer)

while True:
    print("\n1. Ask a question")
    print("2. Exit")
    choice = input("Choose an option: ")

    if choice == "1":
        question = input("Ask your question: ")
        try:
            answer = answer_question(question)
        except Exception as error:
            print(f"Error generating response: {error}")

        print("\nAI Tutor:")
        print(answer)
        
    elif choice == "2":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")