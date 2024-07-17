# Ask My Doc Service

## Introduction
This project is a Python service developed to answer questions based on the content of a specific Word document using FastAPI, llama-index, and OpenAI.

## Setup Instructions

### Prerequisites
- Python 3.12.4
- Docker and Docker Compose

### Installation
1. Clone the repository.
2. Navigate to the project directory.

### Running the Application

#### Using Docker
1. Build and run the application:
    ```sh
    docker-compose up --build
    ```

2. Access the API documentation:
    Open your browser and navigate to `http://127.0.0.1:8000/docs`.

### API Usage
- Endpoint: `/ask`
- Method: `POST`
- Body:
    ```json
    {
      "question": "Your question here"
    }

### Example Questions
- "What is the title of the document?"
- "What is this task designed for?"
- "Can I use other people for this assignment?"


## Contact
For any questions, feel free to contact me.
