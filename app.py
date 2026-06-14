"""
app.py — Gradio Web Interface for The Unofficial Guide

What this script does:
- Launches a local web UI at http://localhost:7860
- User types a question about Alfred University professors
- The system retrieves relevant chunks from ChromaDB and generates a grounded answer
- The answer and source documents are displayed in the UI

Run with: python app.py
"""

import gradio as gr
from query import ask


def handle_query(question):
    """
    Called every time the user submits a question.
    Passes the question to ask() and formats the result for display.
    """
    if not question.strip():
        return "Please enter a question.", ""

    result = ask(question)

    answer = result["answer"]
    sources = "\n".join(f"• {s}" for s in result["sources"])

    return answer, sources


# Build the Gradio UI
with gr.Blocks(title="The Unofficial Guide — Alfred University") as demo:

    gr.Markdown("""
    # The Unofficial Guide — Alfred University
    Ask any question about Alfred University professors based on real student reviews.
    Answers are grounded in collected Rate My Professors data — nothing is made up.
    """)

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Is Joseph Petrillo good for students who struggle with math?",
                lines=2
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False
            )
            sources_output = gr.Textbox(
                label="Retrieved from",
                lines=4,
                interactive=False
            )

    gr.Markdown("""
    ### Example questions to try:
    - Does Lynn Petrillo give good feedback on writing assignments?
    - Is Juliana Gray a harsh grader?
    - Is Joseph Petrillo's calculus class hard?
    - What do students say about Pam Schultz's lectures?
    - Is Sarah Cote a good choice for Writing 1 or Writing 2?
    """)

    # Wire up the button and Enter key to the handler
    submit_btn.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch()
