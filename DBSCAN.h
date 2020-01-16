//
// Created by 123 on 2020/1/16.
//

#ifndef DBSCAN_FACE_DBSCAN_H
#define DBSCAN_FACE_DBSCAN_H
#include<vector>

const int EMB_SIZE = 512;
class Embedding
{
private:
    unsigned int emb_id;
    float emb[EMB_SIZE];
    int cluster_id;
    bool is_key;
    bool visited;
    int label;
    std::vector<unsigned int> arrival_embs;
public:
    Embedding(){};
    Embedding(unsigned int emb_id, std::vector<float>emb, bool is_key);
    unsigned int GetEmbID();
    void SetEmbID(unsigned int emb_id);
    std::vector<float> GetEmb();
    void SetEmb(std::vector<float>emb);
    bool IsKey();
    void SetKey(bool is_key);
    void SetLabel(int label);
    int GetLabel();
    bool IsVisited();
    void SetVisited(bool visited);
    int GetClusterID();
    void SetClusterID(int cluster_id);
    std::vector<unsigned int>& GetArrivalEmbs();

};

class DBSCAN{
private:
    std::vector<Embedding> embs;
    unsigned int emb_size;
    float radius;
    unsigned int emb_num;
    unsigned int min_embs;
    float GetDistance(Embedding emb1, Embedding emb2);
    void SetArrivalEmbs(Embedding& emb);
    void KeyEmbCluster(unsigned int emb_id, int cluster_id);
public:
    DBSCAN(){};
    DBSCAN(char* filename, float radius, int min_embs);
    bool Fit();
    bool Write2File(char* filename);

};





#endif //DBSCAN_FACE_DBSCAN_H
