from transformers import pipeline
import language_tool_python
import gradio as gr

# Grammar correction tool
tool = language_tool_python.LanguageTool('en-US')

# Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Main function for the Smart SOP Coach
def analyze_sop(text):
    # Step 1: Correct grammar
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)

    # Step 2: Generate summary
    summary = summarizer(corrected, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    return corrected, summary

# Gradio UI
iface = gr.Interface(
    fn=analyze_sop,
    inputs=gr.Textbox(lines=20, placeholder="Paste your SOP here", label="Enter SOP"),
    outputs=[gr.Textbox(label="Corrected SOP"), gr.Textbox(label="Summary")],
    title="Smart SOP Coach",
    description="AI-powered app to correct grammar and summarize your Statement of Purpose."
)

iface.launch(share=True)

