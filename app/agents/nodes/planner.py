from app.agents.state import AgentState
from app.gateway.client import get_langchain_llm
import logfire
import re

llm = get_langchain_llm(feature="planner")


# Technical terms that strongly indicate the question may require
# enterprise knowledge-base/documentation retrieval.
TECHNICAL_HINTS = [
    # Kubernetes
    "kubernetes",
    "k8s",
    "pod",
    "pods",
    "deployment",
    "deployments",
    "job",
    "jobs",
    "cronjob",
    "cronjobs",
    "hpa",
    "vpa",
    "configmap",
    "secret",
    "service",
    "services",
    "namespace",
    "namespaces",
    "kubectl",
    "helm",
    "container",
    "containers",

    # Redis / work queue
    "redis",
    "redis-cli",
    "work queue",
    "workqueue",
    "queue",
    "queues",
    "task queue",
    "task queues",

    # Intel
    "intel",
    "sriov",
    "sriov nic",
    "nic",
    "ethernet",
    "xeon",
    "gpu",
    "cpu",

    # Networking
    "network",
    "networking",
    "bgp",
    "ospf",
    "vlan",
    "subnet",
    "routing",
    "router",
    "switch",
    "firewall",
    "tcp",
    "udp",
    "dns",
    "dhcp",

    # Databricks
    "databricks",
    "databricks cli",
    "spark",
    "delta lake",
    "delta",
    "workspace",
    "job run",
    "rest api",

    # Technical/general infrastructure terms
    "cli",
    "command line",
    "command",
    "configuration",
    "configure",
    "config",
    "manifest",
    "architecture",
    "infrastructure",
    "server",
    "servers",
    "cluster",
    "clusters",
    "api",
    "rest api",
    "endpoint",
    "logs",
    "troubleshooting",
    "install",
    "installation",
    "setup",
    "deploy",
    "deployment",
]


def _looks_like_technical_question(message: str) -> bool:
    """
    Deterministic safety net for planner routing.

    If the LLM incorrectly classifies an obvious technical question
    as CONVERSATIONAL, this prevents retrieval from being skipped.
    """
    text = message.lower().strip()

    # Very short conversational messages should remain conversational.
    if len(text.split()) < 3:
        return False

    technical_question_patterns = [
        "how do",
        "how does",
        "how can",
        "how to",
        "what is",
        "what are",
        "what does",
        "why does",
        "why is",
        "where do",
        "where is",
        "which command",
        "which commands",
        "what command",
        "what commands",
        "explain",
        "configure",
        "configuration",
        "setup",
        "install",
        "troubleshoot",
        "troubleshooting",
        "run",
        "create",
        "delete",
        "start",
        "stop",
        "monitor",
    ]

    has_technical_term = any(
        hint in text
        for hint in TECHNICAL_HINTS
    )

    has_technical_pattern = any(
        pattern in text
        for pattern in technical_question_patterns
    )

    return has_technical_term and has_technical_pattern


def planner_node(state: AgentState):
    history = ""

    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = (
        state["messages"][-1]["content"]
        if state["messages"]
        else ""
    )

    prompt = f"""
You are an intelligent Assistant Planner.
Analyze the conversation history and the latest user message.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{user_message}"

Your job is to decide whether the latest message should be handled
from conversation memory or whether it requires retrieval from the
enterprise knowledge base/documentation.

IMPORTANT ROUTING RULE:

A question about the provided documentation, enterprise knowledge
base, uploaded documents, or information that needs to be looked up
MUST go through retrieval.

This includes questions about ANY technology, service, component,
command, procedure, configuration, architecture, troubleshooting
step, or implementation detail that may be described in the
enterprise documentation.

The technology does NOT need to be Kubernetes, Intel, Networking,
or Databricks.

For example, questions about Redis, Redis work queues, Redis CLI,
commands, configuration, task queues, or other technologies described
in the enterprise documentation MUST go through retrieval.

Examples:

"What is this document about?"
"What does this document say?"
"What does the documentation cover?"
"Summarize the document."
"Explain this document."
"What are the main topics in the documentation?"
"What information is available in the knowledge base?"
"What does the knowledge base say about Kubernetes?"
"According to the documentation, how does this work?"
"Find information about Kubernetes in the documentation."
"What does the document say about networking?"
"What does the documentation say about Databricks?"
"How do you fill the Redis work queue with tasks using the CLI?"
"What Redis CLI commands are used in the documentation?"
"How is the Redis work queue configured?"

The Redis question above MUST NOT be classified as
CONVERSATIONAL.

--------------------------------------------------
CONVERSATIONAL
--------------------------------------------------

Classify as CONVERSATIONAL only when the latest message can
reasonably be answered from the conversation itself without
retrieving information from the knowledge base.

Examples:

"Hi"
"Hello"
"My name is Javad"
"Thanks"
"That makes sense"
"What was my name?"
"What did we discuss earlier?"
"What did you just tell me?"
"Can you repeat that?"
"Okay"
"Sounds good"

A follow-up question should be CONVERSATIONAL only when the answer
depends exclusively on information already present in the
conversation.

--------------------------------------------------
TECHNICAL / KNOWLEDGE-BASE RETRIEVAL
--------------------------------------------------

Technical questions should normally go through retrieval when
documentation or factual technical information is required.

This includes, but is not limited to:

1. Kubernetes
2. Intel hardware
3. Enterprise Networking
4. Databricks
5. Redis
6. CLI commands
7. Work queues
8. Services and infrastructure
9. Configuration
10. Troubleshooting
11. APIs
12. Deployment and operational procedures

Examples:

"What is HPA in Kubernetes?"
"How does Kubernetes scheduling work?"
"What is an Intel SRIOV NIC?"
"What is the Databricks CLI command to get a job?"
"When should I use the Databricks REST API?"
"How does BGP work?"
"What is a Kubernetes deployment?"
"How do you fill the Redis work queue with tasks using the CLI?"

For these questions, output a concise refined search query.

--------------------------------------------------
IMPORTANT DISTINCTION
--------------------------------------------------

Do NOT classify a question as CONVERSATIONAL merely because:

- it is short
- it does not contain a specific technology name
- it says "this document"
- it says "the document"
- it says "documentation"
- it says "knowledge base"
- it asks for a summary or explanation
- previous conversation contains related information

If the user is asking about the provided enterprise documentation
or knowledge base, route it to retrieval.

Technical questions that ask HOW, WHAT, WHY, WHICH COMMAND,
CONFIGURATION, SETUP, TROUBLESHOOTING, or IMPLEMENTATION details
should normally be treated as retrieval questions when they concern
enterprise technology or infrastructure.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Output ONLY one of the following:

CONVERSATIONAL

OR

a refined technical/knowledge-base search query.

Do not provide explanations.
Do not provide multiple alternatives.
Do not use markdown.
"""

    with logfire.span("🧭 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")

    # ------------------------------------------------------------
    # DETERMINISTIC TECHNICAL ROUTING FALLBACK
    # ------------------------------------------------------------
    #
    # If the LLM incorrectly calls an obvious technical question
    # "CONVERSATIONAL", force it into retrieval.
    #
    if (
        decision.upper() == "CONVERSATIONAL"
        and _looks_like_technical_question(user_message)
    ):
        decision = user_message.strip()

        logfire.info(
            f"Planner override: technical question routed to retrieval | "
            f"query={decision}"
        )

    # ------------------------------------------------------------
    # CONVERSATIONAL ROUTING
    # ------------------------------------------------------------

    if decision.upper() == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped"
            ]
        }

    # ------------------------------------------------------------
    # RETRIEVAL ROUTING
    # ------------------------------------------------------------
    #
    # Preserve important document-specific identifiers from the
    # original user question.
    #
    # Example:
    #
    # User:
    # "What does the parallelism field do in the Kubernetes
    #  job-wq-2 manifest?"
    #
    # LLM might produce:
    # "Kubernetes Job manifest parallelism field description"
    #
    # We preserve:
    # "job-wq-2"
    #
    # Final query:
    # "Kubernetes Job manifest parallelism field description job-wq-2"
    #
    # This prevents the refined planner query from losing important
    # resource/document identifiers.
    #

    resource_names = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]*-\d+\b",
        user_message
    )

    preserved_identifiers = []

    for name in resource_names:
        if name.lower() not in decision.lower():
            preserved_identifiers.append(name)

    if preserved_identifiers:
        decision = (
            f"{decision} "
            f"{' '.join(preserved_identifiers)}"
        )

        logfire.info(
            f"Planner preserved document-specific identifiers: "
            f"{preserved_identifiers}"
        )

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": [
            "Intent: Technical/Knowledge Base",
            f"Search Term: {decision}"
        ]
    }