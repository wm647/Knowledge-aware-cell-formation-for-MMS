import os
import json
import pandas as pd
import requests
import threading
from typing import Union, List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# from BertBiLSTMCRF.main import predict
from BertTENERCRF.main import predict

# 线程安全打印锁
print_lock = threading.Lock()


class KnowledgeExtractor:
    def __init__(self, model_name: str = "deepseek-r1:32b"):
        self.ollama_url = "http://127.0.0.1:11434/api/chat"
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        self.instruction = ("请从以下文本中提取知识三元组，返回格式严格按照（实体1，关系，实体2），"
                            "如有多个三元组用分号分隔，不要解释。实体中不要带括号。")
                            # 返回Json格式，其中要有两个key，一个是paragraph,一个是triples,知识三元组都存在triples里面，都使用[]括号"
                            #  "注意，表示关系的词，请优先使用如下的词语：包含、位于、按照图纸、"
                            # "需要设备、需要工人、需要工位、需要环境、需要物料")

        # 初始化知识库
        self.knowledge_base_path = "./knowledge_base"
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        self.triplets_file = os.path.join(self.knowledge_base_path, "triplets.json")

    def _call_model_api(self, message_content: str) -> str:
        """调用Ollama API"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message_content}],
            "stream": True
        }
        # print('data_not_json',data)
        try:
            response = requests.post(
                self.ollama_url,
                headers=self.headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                timeout=300
            )
            # print("data_json", json.dumps(data, ensure_ascii=False))
            # print("response", response.text)

            response.raise_for_status()
            return response.text
        except Exception as e:
            with print_lock:
                print(f"\nAPI调用失败: {str(e)}")
            return ""

    def _parse_response(self, response_text: str) -> List[Dict]:
        """解析响应文本为三元组（适配字符列表输入）"""
        triplets = []
        lines = response_text.strip().split("\n")
        think_count = 0
        activate = False
        content = []

        def clean_text(s):
            return s.strip("()（）")

        # 第一步：提取有效内容
        for line in lines:
            try:
                parsed = json.loads(line)
                msg_content = parsed.get("message", {}).get("content", "")

                # 过滤特殊标记
                if msg_content in ["<think>", "</think>", "\u003c/think\u003e"]:
                    think_count += 1
                    continue

                if think_count >= 2 and activate:
                    content.append(msg_content)
                elif think_count >= 2 and not activate:
                    activate = True
            except:
                continue

        # 第二步：合并字符列表为完整字符串
        full_content = "".join([c for c in content if c not in ["\n", " ", ""]])
        print("合并后的完整内容:", full_content)  # 调试输出

        # 第三步：分割并解析三元组
        try:
            # 分割三元组字符串
            triplet_strs = [
                s.strip()
                for s in full_content.split("；")
                if s.strip()
            ]

            for t_str in triplet_strs:
                # 提取括号内容
                if "(" in t_str and ")" in t_str:
                    t_str = t_str[t_str.find("(") + 1: t_str.rfind(")")]
                elif "（" in t_str and "）" in t_str:  # 处理中文括号
                    t_str = t_str[t_str.find("（") + 1: t_str.rfind("）")]

                # 合并字符并分割实体
                parts = []
                current_part = []
                for char in t_str:
                    if char in ["，", ","]:  # 中英文逗号均支持
                        parts.append("".join(current_part).strip())
                        current_part = []
                    else:
                        current_part.append(char)
                # 添加最后一个部分
                if current_part:
                    parts.append("".join(current_part).strip())

                # 验证并保存三元组
                if len(parts) == 3:
                    triplets.append({
                        "subject": clean_text(parts[0]),
                        "relation": clean_text(parts[1]),
                        "object": clean_text(parts[2])
                    })
                    print(f"解析成功: {parts}")  # 调试输出
                else:
                    print(f"无效三元组格式: {t_str}")

        except Exception as e:
            print(f"解析异常: {str(e)}")

        return triplets

    def split_text_to_string(self, text: str) -> str:
        """
        将文本逐字分隔，并用空格连接，返回字符串。
        """
        return " ".join([char for char in text if char.strip() != ""])

    def process_input(self, text: str) -> List[Dict]:
        """处理单条文本输入"""
        #这里对文本进行一个NER，输入给query
        chars = self.split_text_to_string(text)
        # print("字符长度",len(chars))
        # print("字符", chars)
        if len(chars)>256:
            query = f"{self.instruction}\n{text}\n"
            response = self._call_model_api(query)
            return self._parse_response(response) if response else []
        else:
            print("字符长度", len(chars))
            text_label = predict(chars)
            # text_label危一个tuple，第一个[0]值为分词后的原输入，第二个[1]为预测后的BIOES结果
            instruction_BIOES = (
                "请注意，为了方便你理解，我帮你对文本进行了命名体识别划分，你可以根据此来更好理解文本。"
                "该NER采用BIOES标注体系，分别代表实体开头、实体中间、非实体、实体结尾和单词可成为实体，"
                "以B-X的形式标注，其中X的对应如下所示："
                "Pd:整机产品； Fm:功能模块; Co:零部件; Fe:零部件属性，或者表面、端面;"
                "Df:缺陷特性; Pf:工艺文件、操作手册; Pr:工艺、工步; Td1:工装设备;"
                "Ed:电装设备; Wd:焊接设备; Dd:调试仪器; Td2:转运设备，多为拖车、AGV等;"
                "Wo:工人、操作员; Ws:工位、工作站; Ma:物料"

            )

            query = f"{self.instruction}\n{text}\n" + instruction_BIOES + str(text_label[1])
            response = self._call_model_api(query)
            output = self._parse_response(response) if response else []
            # print("This is the response text:",output)
            return output

    def _process_single_row(self, idx: int, text: str) -> tuple:
        """处理单行数据并返回带日志的结果"""
        try:
            # 空内容处理
            if not text.strip():
                return (idx, text, [], "空内容已跳过")

            # 执行处理
            triplets = self.process_input(text)
            print("triplets", triplets)
            print("text", text)

            # 结果验证
            if not triplets:
                return (idx, text, [], "未提取到有效三元组")

            return (idx, text, triplets, "")
        except Exception as e:
            return (idx, text, [], f"处理失败: {str(e)}")

    def process_file(self, file_path: str):
        """处理输入文件"""
        # 读取数据
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            if '工序内容' not in df.columns:
                raise ValueError("EXCEL文件必须包含'工序内容'列")
            data = df['工序内容'].astype(str).tolist()
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError("不支持的文件格式")

        # 数据清洗
        clean_data = [
            (idx, text.strip())
            for idx, text in enumerate(data)
            if text.strip()
        ]
        print(f"有效数据行数: {len(clean_data)}/{len(data)}")

        # 多线程处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._process_single_row, idx, text)
                for idx, text in clean_data
            ]

            # 实时显示处理结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                idx, text, triplets, msg = future.result()

                with print_lock:
                    print(f"\n=== 行 {idx + 1} ===")
                    print(f"内容：{text}")

                    if msg:
                        print(f"状态：{msg}")
                    else:
                        print(f"提取到 {len(triplets)} 个三元组：")
                        for t in triplets:
                            print(f"({t['subject']}, {t['relation']}, {t['object']})")

        # 保存所有结果
        all_triplets = []
        for future in futures:
            idx, text, triplets, _ = future.result()
            all_triplets.extend(triplets)
        self._save_knowledge(all_triplets)

    def _save_knowledge(self, new_triplets: List[Dict]):
        """保存到知识库并去重"""
        existing = []
        if os.path.exists(self.triplets_file):
            with open(self.triplets_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)

        # 去重处理
        seen = {(t["subject"], t["relation"], t["object"]) for t in existing}
        added = [t for t in new_triplets
                 if (t["subject"], t["relation"], t["object"]) not in seen]

        if added:
            updated = existing + added
            with open(self.triplets_file, 'w', encoding='utf-8') as f:
                json.dump(updated, f, ensure_ascii=False, indent=2)

            with print_lock:
                print(f"\n知识库已更新，新增 {len(added)} 条，总量 {len(updated)} 条")

    def show_knowledge(self, num: int = 10):
        """查看知识库"""
        if not os.path.exists(self.triplets_file):
            print("知识库为空")
            return

        with open(self.triplets_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n当前知识库（显示前{num}条）：")
        for i, t in enumerate(data[:num], 1):
            print(f"{i}. ({t['subject']}, {t['relation']}, {t['object']})")


if __name__ == "__main__":
    extractor = KnowledgeExtractor()
    #
    # # 单条测试
    # test_text = "在真空环境下使用电子束焊接机连接钛合金部件和不锈钢支架，工作温度保持在1200℃"
    # print("单条测试结果：")
    # results = extractor.process_input(test_text)
    # for t in results:
    #     print(f"({t['subject']}, {t['relation']}, {t['object']})")
    # 处理文件
    file_path = r"data\（公开版）研究工艺数据V5.0_test.xlsx"
    file_path = r"C:\Users\Admin\Desktop\swift_test\Swift\知识三元组抽取验证数据\label.xlsx"
    if os.path.exists(file_path):
        try:
            extractor.process_file(file_path)
            extractor.show_knowledge()
        except Exception as e:
            print(f"\n处理失败：{str(e)}")
    else:
        print("文件不存在")