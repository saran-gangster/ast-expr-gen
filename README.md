# Discord Counting Bot's bot

Automated counting Bot's bot that allows for complex mathematical expressions. I have designed Abstract Syntax Tree-based expression generator that creates varied and human like math problems which evaluate to the correct next number in the count.

-   **AST-Based Generation:** Generator builds a mathematical expression tree and then renders it to a string for complex and deep expressions.
-   **Safe Evaluation Engine:** A sandboxed `eval()` environment to safely compute the value of other users' expressions fearing of script injections.

This generator works by "reverse-solving" the math. Given a target number (e.g., `42`), it randomly chooses an operation (like addition) and solves for one of the inputs (`42 - 10 = 32`). It then recursively calls itself to generate expressions for `32` and `10`. This process builds a tree of operations, which is then formatted into a string. This method ensures that the final expression, no matter how complex, will always evaluate to the correct target number.

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/ast-expr-gen.git
    cd ast-expr-gen
    ```

2.  **Install Dependencies:**
    ```bash
    pip install aiohttp
    ```

3.  **Create and Fill `config.json`:**
    <br>Copy this content in your newly created config.json

    ```json
    {
      "DISCORD_TOKEN": "YOUR_DISCORD_USER_TOKEN",
      "USER_ID": "YOUR_NUMERIC_USER_ID",
      "CHANNEL_ID": "THE_ID_OF_THE_CHANNEL_TO_COUNT",
      "COUNTING_BOT_ID": "510016054391734273",
      "STATE_FILE": "bot_state.json",
      "GENERATOR_MODE": "AST"
    }
    ```


    *To get IDs, enable Developer Mode in your Discord settings, then right-click a user or channel and select "Copy ID".*
    #####   To Get Your Discord Token
    1.  Open Discord in a web browser (like Chrome or Firefox).
    2.  Open the Developer Tools by pressing **Ctrl+Shift+I** (or **Cmd+Option+I** on Mac).
    3.  Go to the **Network** tab.
    4.  Type `/api` into the filter box to find network requests made to Discord's API.
    5.  Click on any of the entries in the list (e.g., `messages`, `typing`, `science`), If not found, try sending an message in discord anywhere, it should appear now.
    6.  In the panel that appears, go to the **Headers** tab and scroll down to **Request Headers**.
    7.  Find the `authorization` header. The long string of text next to it is your token. Copy it carefully, without any quotes.


## Usage

### Running the Bot

```bash
python main.py
```

---

*Disclaimer: Using a self-bot is against Discord's Terms of Service and may lead to account termination. This project is provided for educational purposes only. Use at your own risk.*
