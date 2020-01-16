//
// Created by 123 on 2020/1/16.
//
#include "DBSCAN.h"
#include <fstream>
#include <sstream>
#include <iosfwd>
#include <math.h>
#include <iostream>

Embedding::Embedding(unsigned int emb_id, std::vector<float> emb, bool is_key) {
    this->is_key = is_key;
    this->emb_id = emb_id;
    for (int i = 0; i < EMB_SIZE; ++i) {
        this->emb[i] = emb[i];
    }
}

int Embedding::GetLabel() {
    return this->label;
}

void Embedding::SetLabel(int label) {
    this->label = label;
}


unsigned int Embedding::GetEmbID() {
    return this->emb_id;
}

void Embedding::SetEmbID(unsigned int emb_id) {
    this->emb_id = emb_id;
}

std::vector<float> Embedding::GetEmb(){
    std::vector<float> tmp(this->emb, this->emb + EMB_SIZE);
    return tmp;
}

void Embedding::SetEmb(std::vector<float> emb) {
    for (int i = 0; i < EMB_SIZE; ++i) {
        this->emb[i] = emb[i];
    }
}
bool Embedding::IsKey() {
    return this->is_key;
}

void Embedding::SetKey(bool is_key) {
    this->is_key = is_key;
}

bool Embedding::IsVisited() {
    return this->visited;
}

void Embedding::SetVisited(bool visited) {
    this->visited = visited;
}

int Embedding::GetClusterID() {
    return this->cluster_id;
}
void Embedding::SetClusterID(int cluster_id) {
    this->cluster_id = cluster_id;
}

std::vector<unsigned int>& Embedding::GetArrivalEmbs(){
    return this->arrival_embs;
}

/*
函数：聚类初始化操作
说明：将数据文件名，半径，领域最小数据个数信息写入聚类算法类，读取文件，把数据信息读入写进算法类数据集合中
参数：
char* fileName;    //文件名
double radius;    //半径
int minPTs;        //领域最小数据个数
返回值： true;    */
DBSCAN::DBSCAN(char* filename, float radius, int min_embs)
{
    this->radius = radius;        //设置半径
    this->min_embs = min_embs;        //设置领域最小数据个数
    this->emb_size = EMB_SIZE;    //设置数据维度
    std::ifstream ifs(filename);        //打开文件
    if (! ifs.is_open())                //若文件已经被打开，报错误信息
    {
        std::cout << "Error opening file";    //输出错误信息
        exit (-1);                        //程序退出
    }

    unsigned long i=0;            //数据个数统计
    while (! ifs.eof() )                //从文件中读取POI信息，将POI信息写入POI列表中
    {
        std::string line;
        std::getline(ifs, line);
//        std::cout<<line<<std::endl;
        std::istringstream sin(line);
        std::string field;
        int j = 0;
        Embedding emb;                //临时数据点对象
        int label = 0;
        std::vector<float> tmp_emb(EMB_SIZE);    //临时数据点维度信息
        while (std::getline(sin, field, ',')){
            if(EMB_SIZE == j){
                label = atoi(field.c_str());

                break;
            }
            tmp_emb[j] = atof(field.c_str());
//            std::cout<<tmp_emb[j]<<std::endl;
            j++;

        }
        emb.SetKey(false);
        emb.SetLabel(label);
        emb.SetEmb(tmp_emb);    //将维度信息存入数据点对象内
        emb.SetEmbID(i);                    //将数据点对象ID设置为i
        emb.SetVisited(false);            //数据点对象isVisited设置为false
        emb.SetClusterID(-1);            //设置默认簇ID为-1
        this->embs.push_back(emb);            //将对象压入数据集合容器
        i++;        //计数+1
    }
    ifs.close();        //关闭文件流
    this->emb_num =i;            //设置数据对象集合大小为i
    for(unsigned long i=0; i<this->emb_num;i++)
    {
        this->SetArrivalEmbs(this->embs[i]);            //计算数据点领域内对象
    }
}

/*
函数：将已经过聚类算法处理的数据集合写回文件
说明：将已经过聚类结果写回文件
参数：
char* fileName;    //要写入的文件名
返回值： true    */
bool DBSCAN::Write2File(char* filename)
{
    std::ofstream of1(filename);                                //初始化文件输出流
    for(unsigned long i=0; i<this->emb_num;i++)                //对处理过的每个数据点写入文件
    {
//        for(int d=0; d<EMB_SIZE ; d++)                    //将维度信息写入文件
//            of1<<this->embs[i].GetEmb()[d]<<",";
        of1 << "ClusterID: "<< this->embs[i].GetClusterID()<< ", Orignal Label:"<< this->embs[i].GetLabel() <<std::endl;        //将所属簇ID写入文件
    }
    of1.close();    //关闭输出文件流
    return true;    //返回
}

/*
函数：设置数据点的领域点列表
说明：设置数据点的领域点列表
参数：
返回值： true;    */
void DBSCAN::SetArrivalEmbs(Embedding& emb)
{
    for(unsigned long i=0; i<this->emb_num; i++)                //对每个数据点执行
    {
        float distance = this->GetDistance(embs[i], emb);    //获取与特定点之间的距离
        if(distance >= radius && i!=emb.GetEmbID())        //若距离小于半径，并且特定点的id与dp的id不同执行
            emb.GetArrivalEmbs().push_back(i);            //将特定点id压力dp的领域列表中
    }
    if(emb.GetArrivalEmbs().size() >= this->min_embs)            //若dp领域内数据点数据量> minPTs执行
    {
        emb.SetKey(true);    //将dp核心对象标志位设为true
        return;                //返回
    }
    emb.SetKey(false);    //若非核心对象，则将dp核心对象标志位设为false
}


/*
函数：执行聚类操作
说明：执行聚类操作
参数：
返回值： true;    */
bool DBSCAN::Fit()
{
    int clusterId=0;                        //聚类id计数，初始化为0
    for(unsigned long i=0; i<this->emb_num;i++)            //对每一个数据点执行
    {
        Embedding& emb = this->embs[i];                    //取到第i个数据点对象
        if(!emb.IsVisited() && emb.IsKey())            //若对象没被访问过，并且是核心对象执行
        {
            emb.SetClusterID(clusterId);                //设置该对象所属簇ID为clusterId
            emb.SetVisited(true);                    //设置该对象已被访问过
            this->KeyEmbCluster(i, clusterId);            //对该对象领域内点进行聚类
            clusterId++;                            //clusterId自增1
        }
        //cout << "孤立点\T" << i << endl;
    }

    std::cout <<"cluster ID" <<clusterId<<std::endl;        //算法完成后，输出聚类个数
    return true;    //返回
}

/*
函数：对数据点领域内的点执行聚类操作
说明：采用递归的方法，深度优先聚类数据
参数：
unsigned long dpID;            //数据点id
unsigned long clusterId;    //数据点所属簇id
返回值： void;    */
void DBSCAN::KeyEmbCluster(unsigned int emb_id, int clusterId)
{
    Embedding& src_emb = this->embs[emb_id];        //获取数据点对象
    if(!src_emb.IsKey())
        return;
    std::vector<unsigned int>& arrval_embs = src_emb.GetArrivalEmbs();        //获取对象领域内点ID列表
    for(unsigned long i=0; i<arrval_embs.size(); i++)
    {
        Embedding& des_emb = this->embs[arrval_embs[i]];    //获取领域内点数据点
        if(!des_emb.IsVisited())                            //若该对象没有被访问过执行
        {
            //cout << "数据点\t"<< desDp.GetDpId()<<"聚类ID为\t" <<clusterId << endl;
            des_emb.SetClusterID(clusterId);        //设置该对象所属簇的ID为clusterId，即将该对象吸入簇中
            des_emb.SetVisited(true);                //设置该对象已被访问
            if(des_emb.IsKey())                    //若该对象是核心对象
            {
                KeyEmbCluster(des_emb.GetEmbID(),clusterId);    //递归地对该领域点数据的领域内的点执行聚类操作，采用深度优先方法
            }
        }
    }
}

//两数据点之间距离
/*
函数：获取两数据点之间距离
说明：获取两数据点之间的欧式距离
参数：
DataPoint& dp1;        //数据点1
DataPoint& dp2;        //数据点2
返回值： double;    //两点之间的距离        */
float DBSCAN::GetDistance(Embedding emb1, Embedding emb2)
{
    float f1dis = 0;
    float f2dis = 0;
    float f1f2dis = 0;
    for (size_t i = 0; i < EMB_SIZE; i++)
    {
        f1f2dis += (float)emb1.GetEmb()[i] * (float)emb2.GetEmb()[i];
        f1dis += (float)emb1.GetEmb()[i] * (float)emb1.GetEmb()[i];
        f2dis += (float)emb2.GetEmb()[i] * (float)emb2.GetEmb()[i];
    }

    float distance = f1f2dis / (sqrt(f1dis) * sqrt(f2dis));
    return distance;        //开方并返回距离
}