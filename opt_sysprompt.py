test_number = 31
# %%
import re
import os, sys
import openai
import math, random
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
from dataclasses import dataclass
import datetime
from time import sleep
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report

load_dotenv("/workspace/zhangmq/optimization-by-prompting/azure.env")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
#print(openai.api_key)

# %%
# let's create some classes that make managing prompts easier
@dataclass
class Solution:
    def __init__(self, solution_name, solution_text, value_name, value):
        self.solution_name = solution_name  # the name of the solution, e.g. "trace" （提示）
        self.solution_text = solution_text  # the text solution to a problem (让我们一步一步来)
        self.value_name = value_name  # the name of the value used to measure the solution, e.g. "length" or "score" （分数）
        if value == -1:  
            value = test_prompt(self.solution_text, test_file_path="./data/train_align_2.xlsx", sample_num=80, iter_id=-1)
            self.value = round(value, 3)
        else:
            self.value = value  # the value of the solution, e.g. 5 or 10 （实际相似度）
        



class MathPrompt:
    def __init__(
        self,
        problem: str,  # text description of the problem to be solved
        instruction: str,  # instructions on what type of solution to provide and in what format
        solution_description: str,  # a description of the solutions and how they are ordered (e.g., "arranged in descending order based on their lengths, where lower values are better")
        example_count: int = 5,  # the maximum number of example solutions to include in the prompt string
        sort_ascending: bool = True,  # whether the solutions are sorted in ascending or descending order
    ):
        self.problem = problem
        self.solution_description = solution_description
        self.solutions = []
        self.instruction = instruction
        self.prompt_string = ""
        self.example_count = example_count
        self.sort_ascending = sort_ascending
        self.ordered_values = [] # the values of the solutions in the order they are determined by the model
        self.solutions_string = ""

    def update_prompt_string(self):
        """
        Creates a string representation of the prompt that can be used to display the prompt to the user or provide it to a language model.
        """
        # create a string representation of the last solution_count solutions
        all_solutions = [f"{solution.solution_name}: {solution.solution_text}\n{solution.value_name}: {solution.value}"
            for solution in self.solutions]
        all_solutions.reverse() # reverse the order so that the solutions with "best" value are first
        
        example_solutions = []
        for i, solution in enumerate(all_solutions):
            if i > self.example_count:
                break
            if solution not in example_solutions:
                example_solutions.append(solution)

        example_solutions.reverse()
        self.solutions_string =  "\n\n".join(example_solutions)
        

        

    def add_solution(self, solution: Solution):
        """
        Adds a solution to the list of solutions, sorts the list by value in ascending or descending order (depending on self.sort_ascending), and updates the prompt string.
        """
        self.solutions.append(solution)

        self.ordered_values.append(solution.value)

        # sort the solutions by value in ascending order
        self.solutions.sort(key=lambda solution: solution.value, reverse=not self.sort_ascending)

        self.update_prompt_string()

    def __repr__(self):
        return self.get_prompt_string()

    def get_prompt_string(self, ):
        examples = self.get_example_qa()
        example_str = ""
        # 打印选择的行
        
        for example in examples:

            example_str+="\n问题："
            example_str+=example[0]
            example_str+="\n标准答案："
            example_str+=example[1]
            example_str+="\n回答："
            example_str+=example[2]
            example_str+="\n人类评分："
            example_str+=str(example[3])
            example_str += "\n"
            
            
        
        return f"{self.problem}\n\n{self.solution_description}\n\n{self.solutions_string}\n\n{self.instruction}\n\n接下来是示例\n{example_str}"


        
    def get_example_qa(self, file_path = "./data/train_align_2.xlsx", question_num = 3):
        df = pd.read_excel(file_path)
        rows = random.sample(range(df.shape[0]), question_num)
        examples = [["" for j in range(df.shape[1])] for i in range(question_num)]
        
        for i, row in enumerate(rows):
            for j in range(df.shape[1]):
                examples[i][j] = df.iloc[row, j]
                
        return examples

# %%

def get_new_prompt_from_gpt(prompt: str, model: str = "gpt-4"): #gpt-3.5-turbo
    completion = openai.ChatCompletion.create(
        engine = model, #gpt-3.5-turbo
        messages=[
        {"role": "system", "content": "You are a helpful assistant in Chinese."},
        {"role": "user", "content": prompt},
        ],
        temperature = 1
    )
    new_prompt = completion.choices[0].message.content
    return new_prompt.replace("<INS>", "").replace("</INS>", "").strip()


def get_new_score_from_gpt(prompt: str, question: str, ground_truth: str, answer: str, model: str): #gpt-3.5-turbo

    message = f"""
        员工提出了如下问题：

        <question>
        {question}
        </question>

        标准答案是：
        <reference>
        {ground_truth}
        </reference>

        实习生给出的回答是：
        <answer>
        {answer}
        </answer>

        首先，请分析实习生的回答，并逐步给出你的分析理由。
        随后，请判断实习生回答能否被接受，并逐步分析给出你的理由。
        最后，按照<score> </score>的格式，给出你的评分。"""
   
    completion = openai.ChatCompletion.create(
        engine = model, #gpt-3.5-turbo
        messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
        ],
        temperature = 0 
    )
    response = completion.choices[0].message.content
    print(response)
    return response


def extract_score(s):
    result = re.findall(r'<score>(.*?)</score>', s, re.DOTALL)
    return int(result[0])

# %%
def calculate_similarity(hyps, refs):
    from rouge import Rouge
    import jieba
    rouge = Rouge()
    hyps = [' '.join(jieba.cut(i)) for i in hyps]
    refs = [' '.join(jieba.cut(i)) for i in refs]
    score = rouge.get_scores(hyps, refs, avg=True)
    return score['rouge-l']['f']

def get_qa_list(file_path):
    dataframe = pd.read_excel(file_path)
    question_list = dataframe.iloc[:, 0].tolist()
    truth_list = dataframe.iloc[:, 1].tolist()
    answer_list = dataframe.iloc[:, 2].tolist()
    human_score_list = dataframe.iloc[:, 3].tolist()
    return question_list, truth_list, answer_list, human_score_list, dataframe

def find_floats_and_ints_in_string(input_string):
    potential_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", input_string)
    return [float(s) if '.' in s else int(s) for s in potential_numbers]

def format_scores(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(0.5 < y_pred , 1, y_pred)
    y_pred = np.where((y_pred <= 0.5) & (y_pred!= -10), 0, y_pred)
    pred_fault_index = np.where(y_pred == -10)
    for i in pred_fault_index[0]:
        if y_true[i]:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    return y_true, y_pred


def binary_cross_entropy(y_true, y_pred):
    
    epsilon = 1e-15  # 极小值，防止计算log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # 将预测值限制在[epsilon, 1-epsilon]区间内
    N = y_true.shape[0]  # 总样本数
    loss = -(1.0/N) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return loss

def MSE_loss(y_true, y_pred):
    return np.mean(np.square(np.array(y_true) - np.array(y_pred)))
def MAE_loss(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
def absolute_loss(y_true, y_pred):
    return np.sum(np.not_equal(np.array(y_true), np.array(y_pred)))

def cross_entropy(y_true, y_pred):
    # 先将预测值限制在一个非常小的区间内, 防止出现无穷大的情况
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    
    # 计算每一个标签值的交叉熵
    ce = - np.sum(y_true * np.log(y_pred)) / len(y_true)
    return ce
def one_hot_encoding(y):
    #classes = np.unique(y)
    classes = np.array([1, 0, 0.5])
    #print(type(classes), classes)
    n_classes = len(classes)
    #classes = 
    one_hot = np.zeros((len(y), n_classes))
    for i, c in enumerate(classes):
        one_hot[y==c, i] = 1
    return one_hot
def CE_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(0.7 <= y_pred , 1, y_pred)
    y_pred = np.where((y_pred <= 0.3) & (y_pred!= -10), 0, y_pred)
    y_pred = np.where((0.3 < y_pred) & (y_pred < 0.7 ), 0.5, y_pred)
    y_true = one_hot_encoding(y_true)
    y_pred = one_hot_encoding(y_pred)
    
    return cross_entropy(y_true, y_pred)

def get_prompt_score_from_model(prompt, file_path, iter_id = -1, sample_num = -1):
    df = pd.read_excel(file_path)
    if sample_num>0:
        df = df.sample(n=sample_num)
    
    question_list = df.iloc[:, 0].tolist()
    truth_list = df.iloc[:, 1].tolist()
    answer_list = df.iloc[:, 2].tolist()
    #
    #question_list, truth_list, answer_list, human_score_list, df = get_qa_list(file_path)
    #print(type(human_score_list[2]))
    #outputs = []
    #for ques, in zip(question_list, truth_list):
        #instruction = ques + "\n" + prompt + "\n" + ref
        #output = generate_output(instruction=instruction, tokenizer=tokenizer, model=model, config=config)
        #print(f"{output}, {datetime.datetime.now().time().strftime('%H:%M:%S')}")
    gpt_score_list, reason_list = calculate_similarity_by_gpt(prompt = prompt, question_list=question_list, truth_list=truth_list, outputs=answer_list)
    
    df['gpt_score'] = gpt_score_list
    df['gpt_reason'] = reason_list
    #df.to_excel(file_path)
    #@print(human_score_list)
    #print(gpt_score_list)
    #"./data/validation_align_1.xlsx"
    #human_score_list = df.iloc[:, 3].tolist()
    result = re.split('[/.]', file_path)
    result_path = "./results/" + result[-2] + f"_{test_number}_{iter_id}." + result[-1]
    df.to_excel(result_path, index=False)
    #y_true, y_pred = format_scores(human_score_list, gpt_score_list)
    #loss = binary_cross_entropy(y_true, y_pred)
    
    #loss = round(loss, 3)
    #if iter_id == -1:
        #print(classification_report(y_true, y_pred))
    return 0
    return loss
    
    
# yanshun学长：gpt3的返回格式不对 也应当给个低分
def calculate_similarity_by_gpt(prompt, question_list, truth_list, outputs):
    scores = []
    reasons = []
    for ques, output, truth in zip(question_list, outputs, truth_list):
        score = -10
        response = ""
        for _ in range(2):
            try:
                response = get_new_score_from_gpt(prompt=prompt, question=ques, ground_truth=truth, answer=output, model="gpt-4")
                score = extract_score(response)
                if type(score) == None:
                    print("没按格式评分")
                    raise TypeError
                    
                elif not 0 <= score <= 2:
                    print("评分范围不在0-2")
                    raise ValueError
                
            except Exception as e:
                print(f"Exception: {e}")
                sleep(5)
            else:
                break
        
        if score == -10:
            print("打什么分？")
        scores.append(score)
        reasons.append(response)
        with open(f"./logs/valid_log_{test_number}.txt", "a+", encoding="utf-8") as f:
            f.write((f"prompt = {prompt},\n question ={ques},\n output = {output},\n truth = {truth},\n score = {score}, reason = {response}\n\n"))
    return scores, reasons

# %%

def solve_optimize_prompt(problem: MathPrompt, max_iters: int = 80, batch_size: int = 8, opt_model = "gpt-4"):
    """
    Solves a prompt optimization problem using GPT-3.5-turbo.

    Args:
    - problem: a MathPrompt object
    - max_iters: the maximum number of iterations to run
    - batch_size: the number of solutions to generate per iteration
    """
    no_improvement_counter = 0
    failure_counter = 0
    api_count = 0
    
    for i in range(max_iters):
        solutions = []
        for j in range(batch_size):
            # 防炸！
            new_prompt = ""
            for _ in range(5):
                try:
                    #print("waiting gpt-4")
                    new_prompt = get_new_prompt_from_gpt(problem.get_prompt_string(), model=opt_model)
                except Exception as e:
                    print(f"Exception: {e}")
                    #print("1234")
                    sleep(15)
                else:  
                    break
            #print("new_prompt: ",new_prompt)
            new_score = get_prompt_score_from_model(new_prompt, file_path = "./data/train_align_2.xlsx", iter_id=i*batch_size+j, sample_num=-1)
            #generate_output(instruction = new_prompt, tokenizer=tokenizer, model=model, config=config)
            #new_length = calculate_trace_length(points, new_trace)
            #new_score = new_score*10
            #new_score = round(new_score, 3)
            #valid_loss = test_prompt(new_prompt)
            #test_loss = test_loss*10
            #valid_loss = round(test_loss, 3) #valid_score: {valid_loss},
            with open(f"./logs/opt_log_{test_number}.txt", "a+", encoding="utf-8") as f:
                f.write(f"train_score: {new_score}, {new_prompt}, iters = {i}, batch = {j}, {datetime.datetime.now().time().strftime('%H:%M:%S')}\n")
            #print(f"valid_score: {new_score}, test_score: {valid_loss}, {new_prompt}")
            if problem.sort_ascending:
                if new_score < problem.solutions[0].value:
                    no_improvement_counter += 1
            else:
                if new_score > problem.solutions[0].value:
                    no_improvement_counter += 1
            api_count += 1

            # if inf or -inf, then count a failure
            if new_score == math.inf or new_score == -math.inf:
                failure_counter += 1
            elif new_prompt in solutions: # if the trace is a duplicate, count it as a failure #这个不太可能 后面想想怎么修改
                failure_counter += 1
                print(f"Generated trace was a duplicate.")
            else: # only add valid solutions
                solutions.append(Solution("prompt", new_prompt, "loss", new_score))

        for solution in solutions: # add all valid solutions from a batch at once
            problem.add_solution(solution)
        
        if no_improvement_counter > max_iters * batch_size / 2:
            print(f"Stopping early after {i} iterations without improvement.")
            break
        

        with open(f"logs/prompt_log_{test_number}.txt", "a+", encoding="utf-8") as f:
            f.write(f"\n******此时的prompt******\n{problem.get_prompt_string()}\n\n")
        
    print(f"There were {failure_counter} failures out of {api_count} API calls.")
    return problem

def test_prompt(test_prompt, test_file_path = "./data/valid_align_2.xlsx", iter_id = -1, sample_num = -1):
    new_score = get_prompt_score_from_model(test_prompt, file_path=test_file_path, iter_id = iter_id, sample_num = sample_num)
    return new_score


# %%
# ok, let's test out the class with an example from Figure 18 the paper

if __name__ == "__main__":
    problem_statement = "你的任务是生成指令<INS>。"
    solution_description = "下面是一些以前的指令及其Loss。指令的Loss为语言模型对于回答的评分与人类评分的量化差距。Loss越小越好"
    instruction = """生成一个与所有上述指令不同且Loss小于所有上述指令的指令<INS>。该指令应以<INS>开头，以</INS>结尾。指令应明确、有效、便于泛用。
    你可以通过以下途径提出更好的指令：
    1.继承上述指令优点，改善上述指令不足
    2.指出示例评分中接受(1分)拒绝(0分)的规律
    3.摘抄示例传递评分规则"""


    solutions = []



    solutions.append(Solution(
        "prompt",
        """""",
        "loss",
        -1,
    ))



    test_problem = MathPrompt(problem_statement, instruction, solution_description, example_count=5, sort_ascending=False)

    for solution in solutions:
        test_problem.add_solution(solution)

    #print(test_problem) # should only show five solutions in the prompt

    test_problem = solve_optimize_prompt(test_problem, opt_model="gpt-4")

    with open(f"logs/opt_log_{test_number}.txt", "a+", encoding="utf-8") as f:
                f.write(f"{test_problem.ordered_values}\n\n")
    plt.plot(test_problem.ordered_values)
    plt.ylabel("Score")
    plt.xlabel("Solution Number")
    plt.show()
    plt.savefig(f"./logs/fig_{test_number}")


'''

#tokenizer, model, config = get_loaded_model()
#_prompt = """"""
#new_score = get_prompt_score_from_model(_prompt)
#print(new_score) #87.496
with open('/workspace/zhangmq/optimization-by-prompting/logs/opt_log_22.txt', 'r', encoding="utf-8") as file:
    # 读取文件内容
    content = file.read()
prompts = re.findall(r', (.*?), iters', content, re.DOTALL)
print(len(prompts))
loss_list = []
for i, prompt in enumerate(prompts):
    if i%8 == 0:
        print(i)

        loss_list.append(test_prompt(prompt, test_file_path="./data/train_align.xlsx", iter_id=i, sample_num=80))
print(loss_list)
loss_str = [str(l) for l in loss_list]
with open('/workspace/zhangmq/optimization-by-prompting/logs/test_log_22.txt', 'w', encoding="utf-8") as file:
    # 读取文件内容
    
    # loss_list
    file.write(', '.join(loss_str))
#prompt=""""""

#CE_22= 请根据0-1系统为AI模型的回答进行公平评估。假如回答准确、有效并清晰地解决了问题，给予1分；如果回答中存在误导、未完全解答问题或离题，给予0分。评估完成后，请详细说明回答的正确性、问题或遗漏之处，以便我们对AI模型作出改进。
#您的任务是对回答是否满足问题的信息要求进行准确判断，并按照以下规则打分：1. 如果回答精准且完整涵盖了问题中所有关键信息，且没有误导或错误信息，分为1。2. 如果回答主要正确，但部分关键信息未提及，分为0.5。3. 如果回答失实或未包含问题的关键信息，分为0。在评分过程中，务必明确标注出回答中正确的、疏忽的和错误的信息，以保障评分的准确无误。
CE_24 = """在评价AI模型的回答时，请按照以下步骤操作：

1.准确性与全面性：如果回答正确且全面地解答了问题，评分为1；如果回答有误或者不全面，评分为0。

2.连贯性与逻辑性：一种流畅、逻辑清晰且符合问题设定的回答则评分为1，反之则为0。

3.是否违反设定：除非被明确要求，否则回答中不应出现“是、不是、可以、不可以”等断定性用语。如符合要求，评分为1；如不符合此要求，则为0。

4.避免直接承诺：回答中不应出现承诺性语言。如果回答中没有承诺，则评分为1；如果回答包含承诺，则评分为0。

5.具体问题的高标准：对于需要具体的数字答案或解答性问题，需要回答准确且明确，如果满足则评分为1，不满足则评分为0。"""
CE_25 = """作为人力资源部的AI助手，您的职责是依据下列原则，对员工的问题提供明确、全面而且易于理解的回答：

1. 对于是或非的问题，需要直接、明确的回答。
2. 当问题需要具体信息，例如查询特定流程的步骤或特定员工的情况，您需要提供完整且准确的信息。
3. 当问题涉及具体的数据或数字时，您需要提供精确的回答。
4. 当需要解释有关公司的政策、程序或规章制度时，您需要提供简明、清晰而且易于理解的回答。

请同时避免以下常见错误：

1. 避免在答案中出现逻辑混乱或自相矛盾。
2. 在非是或非的问题中，避免错误使用“是”或“否”的表述方式。
3. 避免在回答中包含无关的信息或引用，以防答案过长或引起混淆。
4. 在回答员工的需求和请求时，不要提前作出未经确认或未公开的承诺。

回答将根据0-1的标准进行评价。如果答案准确、全面，并能有效地解答问题，评为1；如果答案错误、不能完全解答问题，或者提供了误导性的信息，评为0。在评价结束后，我们会提出答案的优点以及需要改进的地方。"""
prompt = CE_25
print(test_prompt(prompt, test_file_path="./data/valid_align_2.xlsx", sample_num = -1))
'''
# %%

