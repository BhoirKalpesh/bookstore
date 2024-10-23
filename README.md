# Bookstore Chatbot Classification and Data Extraction

This interview involves building a chatbot for an online bookstore that can classify user messages and extract relevant information from them. 

## Submission Instructions

- You are free to use any programming language and libraries to complete the tasks, but Python is preferred.
- Please upload your code to a public repository on GitHub and share the link with us.

## Task Description

### 1. Message Classification

Develop a system that classifies user messages. Available categories:
- **Book Search**: Users looking for information about a specific book.
- **Order Status**: Users asking about their order status.
- **Review Submission**: Users submitting a review for a book.
- **Store Policy Enquiry**: Users asking about store policies (e.g., return policy, shipping policy).

### 2. Data Extraction

Extract relevant information from the messages based on the identified categories. For example:
- **Book Search**: Extract the book title.
- **Order Status**: Extract the order ID.
- **Review Submission**: Extract the review content.
- **Store Policy Enquiry**: Identify the specific policy being inquired about.

### Data Field Descriptions

| Field Description   | Example                                                  | Format                          |
|---------------------|----------------------------------------------------------|---------------------------------|
| Book Title          | The title of the book being inquired about or reviewed.  | Text (any length)               |
| Order ID            | Unique identifier for the order.                         | Alphanumeric, Fixed Pattern: 1 alphabet, 3 numbers, 2 alphabets (e.g. A123BC) |
| Review Content      | Content of the review being submitted.                   | Text (any length)               |
| Store Policy        | Specific store policy being inquired about.              | Text (any length)               |

### 3. Closest Title Match

If a message has been classified into **Book Search**, use the provided `books.json` file to find the closest match to the book title extracted from the message. 

### 4. Closest Publication Year Match

If a user mentions any time-related expressions in their message, find the book in the `books.json` file that has the closest publication year to the mentioned time expression.
+++
### 5. Interface

Build a simple interface which allows us to enter a message and view results for the above tasks.

### Additional Task: Evaluation

No implementation is expected for this section. Consider ways to evaluate the performance of your classification and extraction system. Provide a detailed explanation of the chosen evaluation methods.
