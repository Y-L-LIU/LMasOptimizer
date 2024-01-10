import os
import openai
import math, random
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
from dataclasses import dataclass
from model_generate import generate_output, get_loaded_model

load_dotenv("/workspace/zhangmq/optimization-by-prompting/azure.env")
# let's create some classes that make managing prompts easier
@dataclass
class Solution:
    solution_name: str  # the name of the solution, e.g. "trace" （提示）
    solution_text: str  # the text solution to a problem (让我们一步一步来)
    value_name: str  # the name of the value used to measure the solution, e.g. "length" or "score" （分数）
    value: int  # the value of the solution, e.g. 5 or 10 （实际相似度）


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
         
        return f"{self.problem}\n\n{self.solution_description}\n\n{self.solutions_string}\n\n{self.instruction}\n\n接下来是问题示例\n{examples}"


        
    def get_example_qa(self, file_path = "train_108.xlsx", question_num = 3):
        df = pd.read_excel(file_path)
        rows = random.sample(range(df.shape[0]), question_num)

        examples = ""
        # 打印选择的行
        for i in rows:
            examples+="\n问题：\n"
            examples+=df.iloc[i, 0]
            examples+="\n回答：\n"
            examples+=df.iloc[i, 1]
            
        return examples

# ok, let's test out the class with an example from Figure 18 the paper
problem_statement = "你的任务是生成指令<INS>。"
solution_description = "下面是一些以前的指令及其得分。指令的得分为语言模型产生的回答与真实答案的相似度。相似度越高越好。"
instruction = "生成一个与所有上述<INS>指令不同且得分高于所有上述<INS>指令的指令。该指令应以<INS>开头，以</INS>结尾。该指令应简洁、有效、泛用，并且可以指示模型正确地回答以下的问题。"


solutions = []
"""
solutions.append(Solution(
    "prompt",
    "在回答用户问题时，如果你不知道问题的答案，请回答你不知道。当你对于问题答案不确定时，搜索下面的参考知识。\
    给定你四个相关的参考知识：{refs}，请以叠境公司人事部AI助理的身份，用中文礼貌直接回答下面的问题：{question} ",
    "score",
    0.340,
))

solutions.append(Solution(
    "prompt",
    "回答用户问题时，一定要基于公司的相关政策和规定。如果问题与公司报销、转岗、奖金等方面相关，都可以引用公司的内部文件或政策来给出具体的回答。如果你不确定问题的答案，可以先了解并确认问题的详细信息，然后根据实际情况指导用户去查阅相关的公司文件或者政策。",
    "score",
    0.258,
))
"""
test_problem = MathPrompt(problem_statement, instruction, solution_description, example_count=5, sort_ascending=False)

for solution in solutions:
    test_problem.add_solution(solution)

#print(test_problem) # should only show five solutions in the prompt

# ok, those were close enough; now let's start working on feeding the problems to a model
#load_dotenv("/workspace/zhangmq/optimization-by-prompting/azure.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
print(openai.api_key)
def get_new_prompt(prompt: str, model: str = "gpt-4"): #gpt-3.5-turbo
  completion = openai.ChatCompletion.create(
    engine = model, #gpt-3.5-turbo
    messages=[
      {"role": "system", "content": "You are a helpful assistant in Chinese."},
      {"role": "user", "content": prompt},
    ]
  )
  prompt = completion.choices[0].message.content
  return prompt.replace("<INS>", "").replace("</INS>", "").strip()

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
    answer_list = dataframe.iloc[:, 1].tolist()
    reference_list = dataframe.iloc[:, 2].tolist()
    
    return question_list, answer_list, reference_list

def get_prompt_score_from_model(prompt, tokenizer, model, config, validation_path = "validation_108.xlsx"):
    
    question_list, answer_list, reference_list = get_qa_list(validation_path)
    outputs = []
    for ques, ref in zip(question_list, reference_list):
        instruction = ques + "\n" + prompt + "\n" + ref
        outputs.append(generate_output(instruction=instruction, tokenizer=tokenizer, model=model, config=config))
    score = calculate_similarity(outputs[:15], answer_list[:15]) # 测试用 
    return score
    


# let's make another utility function that generates some random solutions
def generate_random_solutions(points: list, num_solutions: int = 5):
    solutions = []

    

    for i in range(num_solutions): 
        random_ordering = list(range(len(points)))  # convert range to list
        random.shuffle(random_ordering)  # shuffle the list in-place
        trace = f"<trace>{','.join(map(str, random_ordering))}</trace>"  # convert integers to strings before joining
        solutions.append(Solution("trace", trace, "length", (points, trace)))

    return solutions


def solve_optimize_prompt(problem: MathPrompt, m, max_iters: int = 30, batch_size: int = 3, opt_model = "gpt-35-turbo"):
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
    tokenizer, model, config = get_loaded_model()
    for i in range(max_iters):
        solutions = []
        for j in range(batch_size):
            new_prompt = get_new_prompt(problem.get_prompt_string(), model=opt_model)
            
            #(new_prompt, f"iters = {i}, batch = {j}")
            new_score = get_prompt_score_from_model(new_prompt, tokenizer, model, config)
            #new_length = calculate_trace_length(points, new_trace)
            new_score = round(new_score, 3)
            #print(new_score, f"iters = {i}, batch = {j}")
            with open(f"prompts_{m}.txt", "a+", encoding="utf-8") as f:
                f.write(f"\n{new_prompt}")
                f.write(f"   {new_score}   ")
                f.write(" iters = {i}, batch = {j}")



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
                solutions.append(Solution("prompt", new_prompt, "score", new_score))
        


        for solution in solutions: # add all valid solutions from a batch at once
            problem.add_solution(solution)
        
        if no_improvement_counter > max_iters * batch_size / 2:
            print(f"Stopping early after {i} iterations without improvement.")
            break

        with open(f"logs_{m}.txt", "a+", encoding="utf-8") as f:
            f.write(f"\n\此时的prompt：\n{problem.get_prompt_string()}\n\n")
        
    print(f"There were {failure_counter} failures out of {api_count} API calls.")
    return problem

for m in range(100):
    test_problem = MathPrompt(problem_statement, instruction, solution_description, example_count=5, sort_ascending=False)
    solve_optimize_prompt(test_problem, m, opt_model="gpt-4")



