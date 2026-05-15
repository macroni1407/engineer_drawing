import io
def texts_to_txt_buffer(texts):
    """
    texts: list[str]
    """

    txt_buffer = io.BytesIO()

    combined_text = "\n\n".join(texts)

    txt_buffer.write(
        combined_text.encode("utf-8")
    )

    txt_buffer.seek(0)

    return txt_buffer