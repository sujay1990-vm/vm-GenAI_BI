class ConversationManager:
    def __init__(self, history_limit=5):
        """
        Initialize the conversation manager with a history limit.
        """
        self.history = []
        self.history_limit = history_limit

    def update_conversation_history(self, question, answer):
        """
        Store previous questions and answers, but limit the history size.
        """
        self.history.append(f"Question: {question} \n Answer: {answer}")
        if len(self.history) > self.history_limit:
            self.history.pop(0)  # Remove the oldest entry to keep the history within the limit

    def get_conversation_context(self):
        """
        Get the full conversation context as a string.
        """
        return self.history
    
    def clear_history(self):
        """
        Clear the conversation history to erase context.
        """
        self.history = []

    def reformulate_question(self, query_text, llm):
        """
        Reformulate the follow-up question based on the conversation history.
        """
               # Use the reformulation prompt to generate a standalone question
        reformulation_prompt = """
        Given a chat history ( User Questions and LLM response ) and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.

        Chat History:
        {chat_history}

        Latest Question:
        {latest_question}
        """

        # Prepare the input for the LLM model
        formatted_prompt = reformulation_prompt.format(
            chat_history=self.get_conversation_context(),
            latest_question=query_text
        )

        # Get the reformulated question from LLM
        reformulated_response = llm.invoke(input=formatted_prompt)
        standalone_question = reformulated_response.content.strip()

        return standalone_question