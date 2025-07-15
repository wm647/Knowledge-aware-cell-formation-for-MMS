import pandas as pd
import re
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

class RMMS_Graph:
    def __init__(self):
        self.graph = Graph("neo4j://localhost:7687", auth = ("neo4j", "dididaodao"))

        # self.clear_all_nodes()

    # 清除所有节点和关系， 这步不做可能会重复构建节点
    def clear_all_nodes(self):
        query = """
            MATCH (n)
            DETACH DELETE n
        """
        self.graph.run(query)
        print("Cleared all nodes and relationships from the database.")

    def create_graph(self, path):
        self.read_nodes_PBOM_process_method(path)
        self.read_nodes_PBOM_materials()
        self.read_nodes_PBOM_equipment()
        self.read_nodes_PBOM_worker()
        self.update_process_with_work_hours()
        self.read_nodes_PBOM_workstation()

    def graph_query(self, query):
        dup_groups = self.graph.run(query).data()
        return dup_groups

    # 读工艺数据文件，后续可挖掘工序内容
    # 没有调用
    def read_node_process_content(self):
        # 402机
        file_path = r"C:\Users\Admin\Desktop\算法代码\数据\（公开版）研究工艺数据V5.0.xlsx"
        # 笔记本
        # file_path = r"C:\Users\ASUS\Desktop\算法代码\数据\（公开版）研究工艺数据V5.0.xlsx"

        df = pd.read_excel(file_path)

        for index, row in df.iterrows():
            # 检查名称节点是否存在，如果存在则获取该节点，如果不存在则创建新节点
            name_node = self.graph.nodes.match("Product", name=row["名称"]).first()

            if not name_node:
                # 如果“名称”节点不存在，创建新节点
                name_node = Node("Product", name=row["名称"])
                self.graph.create(name_node)
                # print(f"Created new product node: {row['名称']}")
            else:
                # print(f"Product node {row['名称']} already exists.")
                pass

            # 创建工序节点，并记录工时、工序号、工序内容等属性
            process_node = Node("Process",
                                process_id=row["工序号"],
                                process_name=row["工序名称"],  # 根据你表格中的列名调整
                                work_hours=row["人工工时"],  # 根据你表格中的列名调整
                                # work
                                work_content=row["工序内容"])  # 根据你表格中的列名调整
            self.graph.create(process_node)

            # 创建名称节点与工序节点之间的关系
            relationship = Relationship(name_node, "HAS_PROCESS", process_node)
            self.graph.create(relationship)

            # print(f"Created process node with process ID: {row['工序号']} for product {row['名称']}")

    # 读PBOM工艺方法表
    def read_nodes_PBOM_process_method(self, path):
        # 402机
        self.file_path = path
        #self.file_path = r"C:\Users\Admin\Desktop\算法代码\数据\DC调测数据\PBOM数据_51（公开）.xls"
        # 笔记本
        # self.file_path = r"C:\Users\ASUS\Desktop\算法代码\数据\DC调测数据\PBOM数据_51（公开）.xls"
        df = pd.read_excel(self.file_path, sheet_name="PBOM工艺方法表")
        # 遍历每一行数据
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="read_nodes_PBOM_process_method"):
            # 检查 BM号 节点是否已存在，如果不存在则创建新的 BM号 节点
            bm_node = self.graph.nodes.match("Product", name=row["BM号"], version=row["版本号"]).first()
            process_set_node = self.graph.nodes.match("Process_set", name="Process_set"+row["BM号"]).first()
            if not bm_node:
                # 如果 BM号 节点不存在，创建新节点
                bm_node = Node("Product", name=row["BM号"], version=row["版本号"])
                self.graph.create(bm_node)
                # print(f"Created new product node: {row['BM号']}")

            if not process_set_node:
                # 如果 Process_set 节点不存在，创建新节点
                process_set_node = Node("Process_set", name="Process_set"+row["BM号"])
                self.graph.create(process_set_node)
                # print(f"Created new process set node: {row['BM号']}Process_set")

            # 创建工序节点，并记录工序号、工序概述等属性
            process_node = Node("Process",
                                name = row["工序名称"],
                                process_id = row["工序序号"],
                                process_name = row["工序名称"],
                                process_description = row["工序概述"],
                                product_bm = row['BM号'],
                                product_version = row['版本号']
                                )
            self.graph.create(process_node)

            # 创建工序与 BM号 之间的关系
            processed_by_relation = Relationship(bm_node, "PROCESSED_BY", process_set_node)
            has_process_relation = Relationship(process_set_node, "INCLUDES", process_node)

            self.graph.create(processed_by_relation)
            self.graph.create(has_process_relation)

            # 处理前置关联和后置关联的关系
            if pd.notna(row["前置关联"]):
                # 使用正则表达式提取所有方括号内的数字
                pre_processes = re.findall(r'\[(\d+)\]', row["前置关联"])
                for i in pre_processes:
                    # processid默认是int型，这里不对齐的话，下面会导致match不到
                    pre_process = int(i)
                    # 查找前置工序节点
                    pre_process_node = self.graph.nodes.match("Process", process_id=pre_process, product_bm = row['BM号']).first()
                    if pre_process_node:
                        # 创建前置工序关系
                        pre_relation = Relationship(pre_process_node, "PRECEDES", process_node)
                        self.graph.create(pre_relation)
                        # print(f"Created PRECEDES relationship between process {pre_process} and {row['工序序号']}")

    # 读PBOM物料需求表
    def read_nodes_PBOM_materials(self):
        df = pd.read_excel(self.file_path, sheet_name="PBOM物料需求表")
        # for index, row in df.iterrows():
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="read_nodes_PBOM_materials"):
            # 检查 BM号 节点是否已存在，如果不存在则创建新的 BM号 节点
            # bm_node = self.graph.nodes.match("Product", name=row["BM号"], version=row["版本号"]).first() # version有出入，先忽视
            bm_node = self.graph.nodes.match("Product", name=row["BM号"]).first()
            material_set_node = self.graph.nodes.match("Material_set", name="Material_set"+row["BM号"]).first()
            # if not bm_node:
            #     # 如果 BM号 节点不存在，创建新节点
            #     print(f"缺失BM节点: BM号: {row['BM号']}")

            if not material_set_node:
                # 如果 Process_set 节点不存在，创建新节点
                material_set_node = Node("Material_set", name="Material_set"+row["BM号"])
                self.graph.create(material_set_node)
                # print(f"Created new material node: material{row['BM号']}")

            # 创建工序节点，并记录工序号、工序概述等属性
            material_node = self.graph.nodes.match("Material", name = "M_"+str(row["型号"])).first()
            if not material_node:
                material_node = Node("Material",
                                    name = "M_"+str(row["型号"]),
                                    process_id = row["工序序号"],
                                    process_name = row["工序名称"],
                                    type = row["型号"],
                                    quantity = row["数量"],
                                    maximum_quantity_per_set = row["单套最大数量"],
                                    product_bm = row['BM号'],
                                    product_version = row['版本号']
                                    )
                self.graph.create(material_node)

            # process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号'], product_version=row['版本号']).first()
            process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号']).first()


            try:
                has_material_set_relation = Relationship(bm_node, "MATERIAL_REQUIREMENT", material_set_node)
                self.graph.create(has_material_set_relation)

                material_set_includes_relation = Relationship(material_set_node, "INCLUDES", material_node)
                self.graph.create(material_set_includes_relation)

                # 创建物料和工序之间的关系
                requires_material_relation = Relationship(process_node, "REQUIRES_MATERIAL", material_node)
                self.graph.create(requires_material_relation)
            except:
                print("Error in read_nodes_PBOM_materials")

    # 读PBOM仪器需求表
    def read_nodes_PBOM_equipment(self):
        df = pd.read_excel(self.file_path, sheet_name="PBOM仪器需求表")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="read_nodes_PBOM_equipment"):
            # 检查 BM号 节点是否已存在，如果不存在则创建新的 BM号 节点
            # bm_node = self.graph.nodes.match("Product", name=row["BM号"], version=row["版本号"]).first() # version有出入，先忽视
            bm_node = self.graph.nodes.match("Product", name=row["BM号"]).first()
            equipment_set_node = self.graph.nodes.match("Equipment_set", name="Equipment_set"+row["BM号"]).first()
            # if not bm_node:
            #     # 如果 BM号 节点不存在，创建新节点
            #     print(f"缺失BM节点: BM号: {row['BM号']}")

            if not equipment_set_node:
                # 如果 Process_set 节点不存在，创建新节点
                equipment_set_node = Node("Equipment_set", name="Equipment_set"+row["BM号"])
                self.graph.create(equipment_set_node)
                # print(f"Created new material node: material{row['BM号']}")

            # 创建工序节点，并记录工序号、工序概述等属性
            if pd.notna(row['名称']) and row['名称'] != '/':
                equipment_node = self.graph.nodes.match("Equipment", name = str(row["名称"])).first()
                if not equipment_node:
                    equipment_node = Node("Equipment",
                                        name = str(row["名称"]),
                                        process_id = row["工序序号"],
                                        process_name = row["工序名称"],
                                        type = row["类型"],
                                        subdivision = row["细分类"],
                                        quantity = row["数量"] if row["数量"] else 1,
                                        product_bm = row['BM号'],
                                        product_version = row['版本号']
                                        )
                    self.graph.create(equipment_node)

                # process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号'], product_version=row['版本号']).first()
                process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号']).first()


                try:
                    has_equipment_set_relation = Relationship(bm_node, "EQUIPMENT_REQUIREMENT", equipment_set_node)
                    self.graph.create(has_equipment_set_relation)

                    equipment_set_includes_relation = Relationship(equipment_set_node, "INCLUDES", equipment_node)
                    self.graph.create(equipment_set_includes_relation)

                    # 创建物料和工序之间的关系
                    requires_equipment_relation = Relationship(process_node, "REQUIRES_EQUIPMENT", equipment_node)
                    self.graph.create(requires_equipment_relation)
                except:
                    print("Error in read_nodes_PBOM_equipment")

    # 读PBOM人力需求表
    def read_nodes_PBOM_worker(self):
        df = pd.read_excel(self.file_path, sheet_name="PBOM人力需求表")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="read_nodes_PBOM_worker"):
            # 检查 BM号 节点是否已存在，如果不存在则创建新的 BM号 节点
            # bm_node = self.graph.nodes.match("Product", name=row["BM号"], version=row["版本号"]).first() # version有出入，先忽视
            bm_node = self.graph.nodes.match("Product", name=row["BM号"]).first()
            worker_set_node = self.graph.nodes.match("Worker_set", name="Worker_set"+row["BM号"]).first()
            # if not bm_node:
            #     # 如果 BM号 节点不存在，创建新节点
            #     print(f"缺失BM节点: BM号: {row['BM号']}")

            if not worker_set_node:
                # 如果 Process_set 节点不存在，创建新节点
                worker_set_node = Node("Worker_set", name="Worker_set"+row["BM号"])
                self.graph.create(worker_set_node)
                # print(f"Created new material node: material{row['BM号']}")

            # 创建工序节点，并记录工序号、工序概述等属性
            if pd.notna(row['角色']) and row['角色'] != '/':
                worker_node = self.graph.nodes.match("Worker", name = row["角色"]).first()
                if not worker_node:
                    worker_node = Node("Worker",
                                        name = row["角色"],
                                        process_id = row["工序序号"],
                                        process_name = row["工序名称"],
                                        quantity = row["数量"],
                                        product_bm = row['BM号'],
                                        product_version = row['版本号']
                                        )
                    self.graph.create(worker_node)

                # process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号'], product_version=row['版本号']).first()
                process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号']).first()

                try:
                    has_worker_set_relation = Relationship(bm_node, "WORKER_REQUIREMENT", worker_set_node)
                    self.graph.create(has_worker_set_relation)

                    worker_set_includes_relation = Relationship(worker_set_node, "INCLUDES", worker_node)
                    self.graph.create(worker_set_includes_relation)

                    # 创建物料和工序之间的关系
                    requires_worker_relation = Relationship(process_node, "REQUIRES_WORKER", worker_node)
                    self.graph.create(requires_worker_relation)
                except:
                    print("Error in read_nodes_PBOM_worker")

    # 读PBOM工时表
    def update_process_with_work_hours(self):
        df = pd.read_excel(self.file_path, sheet_name="PBOM工时表")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="update_process_with_work_hours"):
            process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号']).first()
            try:
                process_node['machine_hours'] = row['机器工时']
                process_node['labor_hours'] = row['人工工时']
                process_node['standard_work_hours'] = row['准结工时']
                process_node['standard_completion_hours'] = row['准结时间']
                process_node['cycle_time'] = row['周期'] if row['周期'] else row['作业时间']
                self.graph.push(process_node)  # 更新节点数据到图数据库
            except:
                print("Error in update_process_with_work_hours")

    # 读PBOM人力需求表
    def read_nodes_PBOM_workstation(self):
        df = pd.read_excel(self.file_path, sheet_name="PBOM工位需求表")
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="read_nodes_PBOM_workstation"):
            bm_node = self.graph.nodes.match("Product", name=row["BM号"]).first()
            workstation_set_node = self.graph.nodes.match("Workstation_set", name="Workstation_set"+row["BM号"]).first()

            if not workstation_set_node:
                workstation_set_node = Node("Workstation_set", name="Workstation_set"+row["BM号"])
                self.graph.create(workstation_set_node)

            # 创建工序节点，并记录工序号、工序概述等属性
            if pd.notna(row['工位类型']) and row['工位类型'] != '/':
                workstation_node = self.graph.nodes.match("Workstation", name = row["工位类型"]).first()
                if not workstation_node:
                    workstation_node = Node("Workstation",
                                        name = row["工位类型"],
                                        process_id = row["工序序号"],
                                        process_name = row["工序名称"],
                                        type = row["工位属性"],
                                        quantity = row["数量"] if row["数量"] else 1,
                                        product_bm = row['BM号'],
                                        product_version = row['版本号']
                                        )
                    self.graph.create(workstation_node)
                process_node = self.graph.nodes.match("Process", process_id=row["工序序号"], product_bm = row['BM号']).first()

                try:
                    has_workstation_set_relation = Relationship(bm_node, "WORKSTATION_REQUIREMENT", workstation_set_node)
                    self.graph.create(has_workstation_set_relation)

                    workstation_set_includes_relation = Relationship(workstation_set_node, "INCLUDES", workstation_node)
                    self.graph.create(workstation_set_includes_relation)

                    # 创建物料和工序之间的关系
                    requires_workstation_relation = Relationship(process_node, "REQUIRES_WORKSTATION", workstation_node)
                    self.graph.create(requires_workstation_relation)
                except:
                    print("Error in read_nodes_PBOM_workstation")

if __name__ == '__main__':
    path1 = r"C:\Users\Admin\Desktop\算法代码\数据\DC调测数据\PBOM数据_51（公开）.xls"
    path2 = r"C:\Users\Admin\Desktop\算法代码\数据\DC调测数据\PBOM数据-26（公开）.xls"
    path3 = r"C:\Users\Admin\Desktop\博士论文\数据\DC调测数据\新增吊舱调测数据\PBOM数据-2023090924122138.xls"
    g = RMMS_Graph()
    g.clear_all_nodes()
    g.create_graph(path1)
    g.create_graph(path2)
    g.create_graph(path3)

    # g = RMMS_Graph()
    # query = """
    #             MATCH (p:Product)
    #             WITH p.name AS name, p.version AS version, COLLECT(p) AS nodes
    #             WHERE SIZE(nodes) > 1
    #             RETURN name, version, nodes
    #         """
    # query = """
    #             MATCH (p:Product)
    #             RETURN COUNT(p) AS total_products
    #         """
    # output = g.graph_query(query)
    # print(output)







    # data1 = pd.read_excel(path1, sheet_name="PBOM工艺方法表")
    # bm1 = data1["BM号"].unique()
    # data2 = pd.read_excel(path2, sheet_name="PBOM工艺方法表")
    # bm2 = data2["BM号"].unique()
    # data3 = pd.read_excel(path3, sheet_name="PBOM工艺方法表")
    # bm3 = data3["BM号"].unique()
    #
    # print(bm1.tolist()+bm2.tolist()+bm3.tolist())
    # all_bm = bm1.tolist()+bm2.tolist()+bm3.tolist()
    # print(len(all_bm))
    # print(len(list(set(all_bm)))) #共计109型不同产品
    # print(bm1)
    # print(bm2)
    # print(bm3)


#

'''
删除所有节点
MATCH (n)
DETACH DELETE n
'''

