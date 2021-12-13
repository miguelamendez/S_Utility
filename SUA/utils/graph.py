"""Generic Library for working with functional graphs.This library is asummed to by used with pytorch tensors.
    The graphs defined in this library ar directed ciclyc graphs , they only allow one conection betwen two nodes.
    The process function of the graph class only works for directed aciclyc graphs so take that in mind when calling the process function.
"""

#Path for internal libraries
import os
import sys
full_path = os.path.realpath(__file__)
print("graph.py:",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))
from utils.matrices import *
from utils.generic import *

#External libraries
import torch


#Types of graph initialization
#G=Grap() Implemented
#G=Grap(nodes_dic) Implemented
#G=Grap(nodes_dic,edges_dic) Implemented
#----------------------------------------------


class Graph():
    def __init__(self,nodes_dict=[],edges_dict=[]):
        self.nodes=nodes_dict
        self.edges=edges_dict
        self.structure=self.update_structure(self.edges,torch.ones(len(self.edges)))
        #This two lists are for processing the functional graph
        self.processed_edges=dict(list([(node_id,[]) for node_id in self.nodes.keys()]))
        print(self.processed_edges)
        self.processed_struct=torch.zeros(len(self.nodes),len(self.nodes))

    #Node functions:
    def get_nodes_idx(self):
        return list2dict(self.nodes.keys(),keys=True)

    def get_node(self,node_id):
        return self.nodes[node_id]

    def add_node(self,node_id,data):
        #Not working properly
        self.nodes[idx]=new_node
        self.structure=self.update_structure(self.edges,torch.ones(len(self.edges)))
        #self.structure=add_row(self.structure)
        #self.structure=add_column(self.structure)

    def remove_node(self,node_id):
        #Not working properly
        self.nodes.pop(node_id)
        node_edges=self.get_edges(node_id)
        for i in node_edges:
            self.edges.pop(i)
        self.structure=self.update_structure(self.edges,torch.ones(len(self.edges)))
        #self.structure=remove_row(self.structure,node_id)
        #self.structure=remove_column(self.structure,node_id)

    #Edge functions:
    def add_edge(self,edge_tuple,data=[],weight=torch.ones(1)):
        father,son =edge_tuple
        if father not in (self.nodes.keys()):
            raise ValueError("father node doesn't exist")
        if son not in (self.nodes.keys()):
            raise ValueError("son node doesn't exist")
        self.edges[edge_tuple]=data
        self.structure=self.update_structure([edge_tuple],weight,self.structure)

    def remove_edge(self,edge_tuple):
        self.structure=self.update_structure([edge_tuple],torch.zeros(1),self.structure)
        self.edges.pop(edge_tuple)
        return

    def get_parents(self,node_id):
        nodes_idx=self.get_nodes_idx()
        nodes_list=[]
        matrix=torch.transpose(self.structure,0,1)
        vector=matrix[nodes_idx[node_id]]
        for idx, value in enumerate(vector):
            if value!=0:
                nodes_list.append(list(self.nodes.keys())[idx])
        return nodes_list
    
    def get_childs(self,node_id):
        nodes_idx=self.get_nodes_idx()
        nodes_list=[]
        vector=self.structure[nodes_idx[node_id]]
        for idx, value in enumerate(vector):
            if value!=0:
                nodes_list.append(list(self.nodes.keys())[idx])
        return nodes_list
    
    def get_roots(self):
        nodes_list=[]
        matrix=torch.transpose(self.structure,0,1)
        for idx,row in enumerate(matrix):
            if row.sum()==0:
                nodes_list.append(list(self.nodes.keys())[idx])
        return nodes_list
    
    def get_leafs(self):
        nodes_list=[]
        for idx,row in enumerate(self.structure):
            if row.sum()==0:
                nodes_list.append(list(self.nodes.keys())[idx])
        return nodes_list

    def get_edges(self,node_id,childs=False):
        edges_list=[]
        if childs:
            childs=self.get_childs(node_id)
            for i in childs:
                edges_list.append((node_id,i))
        else:
            parents=self.get_parents(node_id)
            for i in parents:
                edges_list.append((i,node_id))
        return edges_list
    
    def get_edges_old(self,node_id):
        nodes_idx=self.get_nodes_idx()
        matrix=torch.transpose(self.structure,0,1)
        vector=matrix[nodes_idx[node_id]]
        edges_list=[]
        for idx, value in enumerate(vector):
            if value!=0:
                edges_list.append((list(self.nodes.keys())[idx],node_id))
        return edges_list

    #Structure functions:
    def update_structure(self,edges,weights,structure=[]):
        if structure==[]:
            n=len(self.nodes)
            structure=torch.zeros(n,n)
        edges_list=[]
        nodes_idx=self.get_nodes_idx()
        for edge in edges:
            father , son = edge
            if father  not in (self.nodes.keys()):
                raise ValueError("father node doesn't exist")
            if son  not in (self.nodes.keys()):
                raise ValueError("son node doesn't exist")
            row=nodes_idx[father]
            column=nodes_idx[son]
            edges_list.append([row,column])
        matrix= update_matrix(structure,edges_list,weights)
        return matrix
    
    #def modify_edge(self,coords,weight=[]):
    #    if bool(weight):
    #        update_struct(self,coords,weight)
    #    else:
    #        update_struct(self,coords,1)


    #Find path functions:

    #Process path
    """Functions for processing functional nodes and edges"""

    def process_v2(self,inputs,new_process=False,leafs=[],children_list=[]):
        #Setting initial parameters
        if new_process:
            self.processed_edges=dict(list([(node_id,[]) for node_id in self.nodes.keys()]))
            leafs=self.get_leafs()
        #Processing input nodes:
        #print("Processing input nodes")
        for node_id in inputs.keys():
            #print("\t Processing node",node_id)
            node=self.nodes[node_id]
            node["outputs"]=node["model"](inputs[node_id])
            #print("\t Node outputs:",node["outputs"])
            #Processing outwards edges:
            #print("\t \t Processing outwards edges")
            if node_id not in leafs: #Check that node is not leaf
                node_edges=self.get_edges(node_id,childs=True)
                for edge_id in node_edges:
                    #print("\t \t \t Processing edge",edge_id)
                    parent_id, child_id = edge_id
                    edge=self.edges[edge_id]
                    edge["output"]=edge["model"](node["outputs"])
                    #print("\t \t \t edge output",edge["output"])
                    self.processed_edges[child_id].append(edge_id)
                    #self.processed_struct=update_structure([edge_id],[1],structure=self.processed_struct)
                children_list=children_list+self.get_childs(node_id)
                children_list = list(dict.fromkeys(children_list))
        #Processing children
        print("child_list",children_list)
        if children_list==[]:
            return
        new_inputs={}
        #print("Processing children")
        #print("children_list:",children_list)
        for node_id in children_list:
            edges=self.get_edges(node_id)
            #print(edges)
            #print(self.processed_edges[node_id])
            #This condition checks if we have all inputs necessary for the children if so we append it to the new inputs
            if sorted(edges)==sorted(self.processed_edges[node_id]):
                new_inputs[node_id]=[]
                for edge_id in edges:
                    #print(edge_id)
                    edge=self.edges[edge_id]
                    new_inputs[node_id].append(edge["output"])
                children_list.remove(node_id)
        print("new_inputs",new_inputs)
        #print("children_list:",children_list)
        #Processing new_inputs
        if new_inputs!={}:
            self.process_v2(inputs=new_inputs,new_process=False,leafs=leafs,children_list=children_list)
        return
                
            
                
            
    #def process(self,inputs,nodes=[]):
    #    if nodes==[]:
    #        nodes=self.get_roots()
    #    leafs=self.get_leafs()
    #    childs_list=[]
    #    for node_id in nodes:
    #        node=self.nodes[node_id]
    #        node["outputs"]=node["model"](inputs[node_id])
    #        if node_id not in leafs:
    #            edges=self.get_edges(node_id,childs=True)
    #            for edge_id in edges:
    #                parent ,child=edge_id
    #                if child not in childs_list:
    #                    childs_list.append(child)
    #                edge=self.edges[edge_id]
    #                edge["output"]=edge["model"](node["outputs"])
    #    new_inputs={}
    #    for i in childs_list:
    #        edges=self.get_edges(i)
    #        for j in edges:
    #            edge=self.edges[edge_id]
    #            new_inputs[i].append(edge["outputs"])
    #    self.process(,nodes=childs_list)
    #        
    #    print(edges)
    #    edges_list=self.edges_list(edges)
    #    parents=[]
    #    for i in edges_list:
    #        parents.append(self.nodes[i[0]])
    #    return parents

    def process_edges(self,inputs,edges):
        return
        
#Example:
nodes_list=["a","b","c","d","e"]
#nodes_list=["a","b","c"]
print(nodes_list)
nodes_dict=build_1val_dict(nodes_list,{"outputs":[],"model":lambda x:(1+torch.tensor(x))})
print(nodes_dict)
edges_list=[("a","b"),("a","c"),("a","d"),("a","e"),("b","c"),("b","d"),("b","e"),("c","d"),("c","e"),("d","e")]
print(edges_list)
edges_dict={("a","b"):{"weight":[],"output":[],"model":lambda x:2+torch.tensor(x).sum()},("a","c"):{"weight":[],"output":[],"model":lambda x:2+torch.tensor(x).sum()},("b","c"):{"weight":[],"output":[],"model":lambda x:2+torch.tensor(x).sum()}}
edges_dict=build_1val_dict(edges_list,{"weight":[],"output":[],"model":lambda x:2+torch.tensor(x).sum()})
print(edges_dict)
graph=Graph(nodes_dict,edges_dict)
print(graph.structure)
#print(graph.edges.keys())
#graph.remove_edge(("a","c"))
#print(graph.edges.keys())
#graph.add_edge(("a","c"),{"weight":[],"output":[],"model":lambda x:2+torch.tensor(x).sum()})
#print(graph.edges.keys())
print(graph.get_edges("c"))
print(graph.get_edges("b"))
print(graph.get_edges("a",childs=True))
inputs={"a":torch.tensor([1])}
graph.process_v2(inputs,True)

for node_id in graph.nodes.keys():
    node=graph.nodes[node_id]
    print(node_id,node["outputs"])
for edge_id in graph.edges.keys():
    edge=graph.edges[edge_id]
    print(edge_id,edge["output"])
print(graph.get_parents("c"))
print(graph.get_childs("a"))
print(graph.get_roots())
print(graph.get_leafs())
