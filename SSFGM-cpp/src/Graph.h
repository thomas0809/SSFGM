#include <cstdio>
#include <vector>
#include <utility>
using namespace std;

class Graph {
public:
	int N, M;
	vector< pair<int,int> > *inlist, *outlist;
	vector< pair<int,int> > edges;

	Graph(int N_) : N(N_), M(0) {
		inlist = new vector< pair<int,int> >[N];
		outlist = new vector< pair<int,int> >[N];
		edges.clear();
	}
	~Graph() {
		delete[] inlist;
		delete[] outlist;
	}

	void add_edge(int u, int v, int type) {
		if (u >= N || v >= N)
			return;
		outlist[u].push_back(make_pair(v, type));
		inlist[v].push_back(make_pair(u, type));
		edges.push_back(make_pair(u, v));
	}
};