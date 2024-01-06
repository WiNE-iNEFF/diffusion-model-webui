from modules import *

demo_page = True

device = ("mps"
    if torch.backends.mps.is_available() else "cuda"
    if torch.cuda.is_available() else "cpu")

demo = gr.Blocks(title='diffusion-model-webui')
with demo:
    gr.HTML("""
            <div style="display: flex; margin-bottom: 0.5%; justify-content: center;">
              <p style="font-size: 24px"><b>diffusion-model-webui</b></p>
             </div>
            """
           )
    with gr.Tabs():
      with gr.TabItem("About"):
        gr.Markdown(f'''
                    # Diffusion Model WebUI
                    ## This is demo space.
                    For start work please edit "app.py" variable "demo_page = False".

                    An educational project that is created to test my own skills and make it easier to work with the ***Diffusers*** HuggingFace library.
                    Full code in <a href='https://github.com/WiNE-iNEFF/diffusion-model-webui'>GitHub page</a>.

                    This gradio space run {'using **Google Colab** with **' + device if RunningInCOLAB else 'with **' + device}**.
                    ''')
        with gr.Accordion("Last update:", open=False):
          gr.Markdown('''
                      05.01.24
                      > The functionality of the '**Fine-tune model**' - '**Unconditional Diffusion Model**' tab is completely ready and tested.
                      ''')
          
        with gr.Accordion("Plans for future updates:", open=False):
          gr.Markdown('''
                      > Completely complete the functionality of the '**Train model**' and '**Fine-tune model**' tabs
                      ''')
          
        with gr.Accordion("Author:", open=False):
          gr.Markdown('''
                      Invented and implemented this project **Artsem Holub (WiNE-iNEFF)**.

                      All my pages:
                      <a href='https://huggingface.co/WiNE-iNEFF'>HuggingFace</a>,
                      <a href='https://github.com/WiNE-iNEFF'>GitHub</a>,
                      <a href='https://twitter.com/wine_ineff'>X (later Twitter)</a>.
                      ''')


      with gr.TabItem("Train model"):
        with gr.TabItem("Unconditional Diffusion Model"):
          future_update()
        with gr.TabItem("Class-conditional Diffusion Model"):
          future_update()


      with gr.TabItem("Fine-tune model"):
        with gr.TabItem("Unconditional Diffusion Model"):
          with gr.Column():
            with gr.Row():
              huggingface_write_token = gr.Textbox(lines=1, label="Huggingface access token write permission (required)", scale=2)
              hf_token = gr.Button("Login in HF", size='sm', scale=1)
              hf_token.click(fn=hf_login, inputs=huggingface_write_token, outputs=None)

            with gr.Accordion("Wandb setting (optional)", open=False):
              wandb_write_token = gr.Textbox(lines=1, label="WanDB api token (optional)")
              with gr.Row():
                wandb_project_name = gr.Textbox(lines=1, label="Project name", placeholder='SimpleProject1')
                wandb_run_name = gr.Textbox(lines=1, label="Run name", placeholder='run 22.06.2001')

              wandb_token = gr.Button("Login in WanDB")
              wandb_token.click(fn=wandb_setup, inputs=[wandb_write_token, wandb_project_name, wandb_run_name], outputs=None)

            with gr.Accordion("Dataset setting (required)", open=False):
              with gr.Row():
                pretrained_model = gr.Textbox(lines=1, label="Pretrained model from HuggingFace model library", placeholder='WiNE-iNEFF/Minecraft-Skin-Diffusion-V2')
                dataset_name = gr.Textbox(lines=1, label="Dataset link from HuggingFace dataset library", placeholder='WiNE-iNEFF/kuvshinov_art_dataset')

              with gr.Row():
                image_size = gr.Textbox(lines=1, value="(64, 64)", label="image size", placeholder='(64, 64)')
                batch_size = gr.Number(value=64, label="Batch size", show_label=True, precision=0)
                color_value = gr.Dropdown(choices=['RGBA', 'RGB'], value='RGB', label="Color value (RGBA/RGB). Must match the values with pretrained_model")

            with gr.Accordion("Train setting (required)", open=False):
              with gr.Row():
                epoch = gr.Number(value=10, label="Number of train epoch", show_label=True, precision=0)
                grad_accumulation_steps = gr.Number(value=2, label="Gradient accumulation steps", show_label=True, precision=0)
              with gr.Row():
                save_to_hub = gr.Number(value=1, label="How many epoch step need to save model checkpoint", show_label=True, precision=0)
                image_example = gr.Number(value=5, label="How many epoch step need to show image example (work only with wandb)", show_label=True, precision=0)

              model_name = gr.Textbox(lines=1, label="Model name to push in hub (required)", placeholder='Minecraft-Skin-Diffusion-V2')
              model_description = gr.Textbox(lines=3, label="Model description (optional)")

          finish_gallery = gr.Gallery()
          start_btn = gr.Button("Start training", variant='primary')

          if demo_page:
          	gr.Info('This is demo page. For start work please edit "app.py" variable "demo_page = False".')
          else:
          	start_btn.click(fn=start_setup,
                          inputs=[pretrained_model,
                                  dataset_name,
                                  image_size,
                                  batch_size,
                                  color_value,
                                  epoch,
                                  grad_accumulation_steps,
                                  save_to_hub,
                                  image_example,
                                  model_name,
                                  model_description],
                          outputs=finish_gallery)

        with gr.TabItem("Class-conditional Diffusion Model"):
          future_update()


      with gr.TabItem("Test model"):
        future_update()

demo.queue().launch(debug=True)