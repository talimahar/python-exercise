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

## Answer to Assignment Questions

1. **What type of questions worked? What questions did not work?**
    - **Worked**: Questions directly related to the document content, e.g., "What is the title of the document?".
    - **Did not work**: General knowledge questions not related to the document context, e.g., "How many states are in the US?".

2. **What improvements should be made to the service to support the non-answered questions?**
    - Enhancing the document indexing and retrieval algorithms.
    - Implementing advanced natural language processing techniques to better understand the context.

3. **Design and architecture to support parallel loading of 100M documents.**
    - Implement a distributed architecture using technologies like Apache Kafka for message queuing and Apache Spark for processing.
    - Use scalable storage solutions like AWS S3 and distributed databases like Cassandra or MongoDB.
    - Implement parallel processing using worker nodes and load balancers.

## Contact
For any questions, feel free to contact me.
