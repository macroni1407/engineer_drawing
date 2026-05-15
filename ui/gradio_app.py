import gradio as gr

def create_demo(process_fn):

    demo = gr.Interface(
        fn=process_fn,

        inputs=gr.Image(
            type="filepath",
        ),

        outputs=[
            gr.Image(
                type="numpy",
                label="Prediction",
            ),

            gr.Gallery(
                label="Cropped Objects",
                columns=3,
            ),

            gr.JSON(),

            gr.File(),
        ],

        title="Engineer Drawing",
    )

    return demo