import os
from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpEvmServerWalletProvider,
    CdpEvmServerWalletProviderConfig,
    cdp_api_action_provider,
    erc20_action_provider,
    pyth_action_provider,
    wallet_action_provider,
    weth_action_provider,
)
from coinbase_agentkit_langchain import get_langchain_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Set your environment variables or replace with your actual values
CDP_API_KEY_ID = os.environ.get("CDP_API_KEY_ID")
CDP_API_KEY_SECRET = os.environ.get("CDP_API_KEY_SECRET")
CDP_WALLET_SECRET = os.environ.get("CDP_WALLET_SECRET")
NETWORK_ID = os.environ.get("NETWORK_ID", "base-sepolia")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. Set up the Coinbase CDP EVM Server Wallet Provider
wallet_provider = CdpEvmServerWalletProvider(CdpEvmServerWalletProviderConfig(
    api_key_id=CDP_API_KEY_ID,
    api_key_secret=CDP_API_KEY_SECRET,
    wallet_secret=CDP_WALLET_SECRET,
    network_id=NETWORK_ID,
))

# 2. Create AgentKit instance with wallet and action providers
agentkit = AgentKit(AgentKitConfig(
    wallet_provider=wallet_provider,
    action_providers=[
        cdp_api_action_provider(),
        erc20_action_provider(),
        pyth_action_provider(),
        wallet_action_provider(),
        weth_action_provider(),
    ],
))

# 3. Set up LangChain tools and LLM
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", openai_api_key=OPENAI_API_KEY)
tools = get_langchain_tools(agentkit)

# 4. Create the LangChain agent
agent = create_react_agent(
    llm=llm,
    tools=tools
)

# Example: Run a simple prompt
response = agent.invoke({"input": "What is the current price of Bitcoin?"})
print(response)
