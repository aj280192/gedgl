import random
import argparse
import logging

def read_edge(edge_list):
    edges = {}
    nodes = []
    with open(edge_list) as l:
        for line in l.readlines():
            src, dst = line.strip().split(' ')
            if (src,dst) not in edges:
                edges[(src,dst)] = 1
            if (dst,src) not in edges:
                edges[(dst,src)] = 1
            
            nodes.append(src)
            nodes.append(dst)
    
    nodes = list(set(nodes))
    return edges, nodes

def neg_genarate(edges,nodes,ratio):
    neg_count = int(len(edges) * ratio)
    node_len = len(nodes)
    neg_edges = []
    
    for _ in range(neg_count):
        trail = 0
        all_flag = False
        while True:
            x = nodes[random.randint(0,node_len-1)]
            y = nodes[random.randint(0,node_len-1)]
            trail += 1
            if trail >= 1000:
                all_flag = True
                break
            if x != y and (x,y) not in edges and (y,x) not in edges:
                neg_edges.append((x,y))
                break
        if all_flag:
            break
    
    return neg_edges

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--edgefile", type=str, required=True,
            help="embedding file path")
    parser.add_argument("--negedgefile", type=str, required=True,
            help="edge list file path")
    parser.add_argument("--negative", type=float, default=0.3,
            help="percentage of negative samples!!")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s')
    
    edges, nodes = read_edge(args.edgefile)
    print('going to genarate negative edges!!')
    neg_edges = neg_genarate(edges,nodes,args.negative)
    
    with open(args.negedgefile,'w') as n:
        for neg in neg_edges:
            n.write(neg[0] + ' ' + neg[1] + '\n')