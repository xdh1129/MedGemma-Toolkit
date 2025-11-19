# BDA-Project

This project is a full-stack web application consisting of a React-based frontend and a Python backend. It leverages Docker for containerization and uses Ollama for running local large language models.

## Prerequisites

Before you begin, ensure you have the following installed on your workstation:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Ollama](https://ollama.ai/)

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd NL_Project
    ```

2.  **Run the start script:**
    The `start.sh` script will handle the following:
    - Check for necessary dependencies.
    - Start a local Ollama server if one is not already running.
    - Prime the required language models.
    - Build and start the frontend and backend services using Docker Compose.

    To start the application, run:
    ```bash
    ./start.sh
    ```

3.  **Accessing the Application:**
    - The frontend is accessible at `http://localhost:8080`.
    - The backend is accessible at `http://localhost:8000`.

## Accessing the Frontend from Your Local Machine

If the application is running on a remote workstation, you can access the frontend in your local browser using SSH port forwarding.

1.  **Open a terminal on your local machine.**

2.  **Run the following SSH command:**
    Replace `user` with your username and `workstation-ip-address` with the IP address of your remote workstation.

    ```bash
    ssh -L 8080:localhost:8080 user@workstation-ip-address
    ```
    This command forwards port 8080 from your local machine to port 8080 on the remote workstation.

3.  **Open your local web browser.**
    Navigate to `http://localhost:8080`. You should now see the application's frontend.

## Services

- **`frontend`**: A React application built with Vite (see `chat-frontend/`), served by Nginx.
  - **Port:** `8080` on the host.
- **`backend`**: A Python application.
  - **Port:** `8000` on the host.
  - This service communicates with an Ollama server running on the host machine.

## Language Models

The backend service is configured to use the following Ollama models:

- `hf.co/unsloth/medgemma-27b-it-GGUF:Q4_K_M`
- `hf.co/mradermacher/II-Medical-8B-GGUF:Q4_K_M`

The `start.sh` script will attempt to prime these models to ensure they are ready to serve requests.
