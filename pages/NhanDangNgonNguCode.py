import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizer, TextClassificationPipeline

st.set_page_config(page_title="Nhận diện Ngôn ngữ lập trình", page_icon="🕵🏽", initial_sidebar_state="auto")

code_snippets_list = [
    """
    def square(value):
        return square**2
    """.lstrip(),
    """
    package main
    import 'fmt'
    func main() {
    	name := name()
	    fmt.Println(name)
    	age := age()
	    fmt.Println(age)
    	height := height()
	    fmt.Println(height)
    }
    func name() string {
	    return "Van Sanh"
    }
    func age() int {
	    return 25
    }
    func height() float32 {
	    return 5.10
    }
    """.lstrip(),
    """
    <?php
        function reverse_string($str1){
            $n = strlen($str1);
            if($n == 1){
                return $str1;
                }
            else{
                $n--;
                return reverse_string(substr($str1,1, $n)).substr($str1, 0, 1);
            }
        }
        print_r(reverse_string('1234')."\n");
    ?>
    """.lstrip(),
    """
    function fibonacci(num){   
        if(num==1)
            return 0;
        if (num == 2)
            return 1;
        return fibonacci(num - 1) + fibonacci(num - 2);
    }
    """.lstrip(),
    """
    def mutate(array)
        array.pop
    end
    """.lstrip()
]

code_snippets = {
    "Python": code_snippets_list[0],
    "Go": code_snippets_list[1],
    "PHP": code_snippets_list[2],
    "Javascript": code_snippets_list[3],
    "Ruby": code_snippets_list[4]
}

st.cache(show_spinner=True)
def load_pipeline():
    pipeline = TextClassificationPipeline(
        model=RobertaForSequenceClassification.from_pretrained("huggingface/CodeBERTa-language-id"),
        tokenizer=RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-language-id")
    )
    return pipeline


st.header("Phát hiện ngôn ngữ Lập trình 🔎")
st.subheader("Dán code vào khung dưới hoặc chọn đoạn trích mẫu 👇🏽")

selected_code_snippet = st.selectbox(
    "Chọn đoạn code ví dụ",
    code_snippets.keys()
)

text = st.text_area(
    label="", 
    value=code_snippets[selected_code_snippet],
    height=250,
    max_chars=None,
    key=None
)

button = st.button("Phát hiện ngôn ngữ 🔎")

if button:
    if not len(text) == 0:
        pipeline = load_pipeline()
        result_dict = pipeline(text)[0]
        #st.write(result_dict["label"].capitalize())
        st.markdown(f"Này là ngôn ngữ **_:red[{result_dict['label'].capitalize()}]_** nè! :sunglasses:")
    else:
        st.markdown("Viết thêm code vào đi!")