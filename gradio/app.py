import gradio as gr

from index import upload_files, get_uploaded_files, delete_files, process_upload
from query import chat

ICON_PATH = "assets/favicon.png"

theme = gr.themes.Monochrome(
    primary_hue="neutral",
    text_size="md",
    spacing_size="md",
    radius_size="md",
    # font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

light_js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

custom_css = """
.scrollable-file-list {
    min-height: 480px;
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #e2e2e2;
    border-radius: 4px;
    padding: 10px;
}
.scrollable-file-list .wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.scrollable-file-list .wrap > div {
    padding: 8px;
    border-bottom: 1px solid #f0f0f0;
}
.scrollable-file-list .wrap > div:last-child {
    border-bottom: none;
}
"""

with gr.Blocks(
        title="PNOC RAG LLM Portal",  
        theme=theme,
        fill_height=True,
        css=custom_css,
        # js=light_js, # Forces light mode
    ) as app:

    with gr.Row(height=100, equal_height=True):
        gr.Image(
            ICON_PATH,
            height=100,
            width=100,
            min_width=100,
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            container=False,
            interactive=False,
            scale=0,
        )
        gr.HTML(
            "<h1 style='font-size: 2.5rem; display: flex; align-items: center; height: 100px;'>PNOC Document RAG Portal</h1>",
            padding=False,
        )
    
    with gr.Tabs():
        with gr.TabItem("Document Upload"):
            with gr.Row(scale=1, equal_height=True):
                with gr.Column():
                    gr.Markdown("### Add files to index")
                    with gr.Row(scale=1, height=500):
                        file_input = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".doc", ".docx"],
                            show_label=False,
                            height=480,
                        )
                    with gr.Row(scale=0):
                        with gr.Row():
                            submit_btn = gr.Button("Upload Files", variant="primary")

                with gr.Column():
                    gr.Markdown("### Files already indexed")
                    with gr.Row(scale=1, height=500):
                        file_list = gr.CheckboxGroup(
                            value=[],
                            interactive=True,
                            show_label=False,
                            scale=1,
                            elem_classes=["scrollable-file-list"], # Invokes custom CSS
                        )
                    with gr.Row(scale=0):
                        refresh_btn = gr.Button("Refresh File List")
                        delete_btn = gr.Button("Delete Selected Files", variant="stop")

            with gr.Row(scale=0):
                output = gr.Textbox(label="Operation Result", lines=2)
                        
            # Initial load of uploaded files
            app.load(
                fn=lambda: gr.CheckboxGroup(choices=get_uploaded_files()),
                outputs=file_list,
                queue=False
            )
            
            # Button actions
            submit_btn.click(
                fn=process_upload,
                inputs=file_input,
                outputs=[output, file_list]
            ).then(
                fn=lambda: gr.CheckboxGroup(choices=get_uploaded_files()),
                outputs=file_list
            )
            
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_uploaded_files()),
                outputs=file_list
            )
            
            delete_btn.click(
                fn=delete_files,
                inputs=file_list,
                outputs=[output, file_list]
            ).then(
                fn=lambda: gr.CheckboxGroup(choices=get_uploaded_files()),
                outputs=file_list
            )
        
        with gr.TabItem("Chat"):
            chat_interface = gr.ChatInterface(
                fn=chat,
                type="messages",
                save_history=True,
                fill_height=True,
                chatbot=gr.Chatbot(
                    type="messages",
                    allow_tags=["think"],
                    height=600,
                ),
                examples=[
                    'What is the current status of renewable energy in the Philippines?',
                    'What are the goals for renewable energy policies?',
                    'What are the difficulties/challenges with implementing renewable energy?'
                ],
            )

if __name__ == "__main__":
    app.launch(
        favicon_path=ICON_PATH,
        server_name="0.0.0.0",
        server_port=7860,
    )
