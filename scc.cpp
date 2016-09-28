
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <string>

struct graph {
  int num_verts;
  int num_edges;
  int* out_array;
  int* out_degree_list;
  int* in_array;
  int* in_degree_list;
} ;
#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) &g->out_array[g->out_degree_list[n]]
#define in_degree(g, n) (g->in_degree_list[n+1] - g->in_degree_list[n])
#define in_vertices(g, n) &g->in_array[g->in_degree_list[n]]


using namespace std;

int rank, size;

void read_edge(char* filename,
  int& num_verts, int& num_edges,
  int*& srcs, int*& dsts)
{
  ifstream infile;
  string line;
  infile.open(filename);

  getline(infile, line, ' ');
  num_verts = atoi(line.c_str());
  getline(infile, line);
  num_edges = atoi(line.c_str());

  int src, dst;
  int counter = 0;

  srcs = new int[num_edges];
  dsts = new int[num_edges];
  for (unsigned i = 0; i < num_edges; ++i)
  {
    getline(infile, line, ' ');
    src = atoi(line.c_str());
    getline(infile, line);
    dst = atoi(line.c_str());

    srcs[counter] = src;
    dsts[counter] = dst;
    ++counter;
  }
  infile.close();
}

void create_csr(int num_verts, int num_edges, 
  int* srcs, int* dsts,
  int*& out_array, int*& out_degree_list,
  int*& in_array, int*& in_degree_list)
{
  out_array = new int[num_edges];
  in_array = new int[num_edges];
  out_degree_list = new int[num_verts+1];
  in_degree_list = new int[num_verts+1];
  int* temp_counts = new int[num_verts];

  for (int i = 0; i < num_edges; ++i)
    out_array[i] = 0;
  for (int i = 0; i < num_verts+1; ++i)
    out_degree_list[i] = 0;
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = 0;

  for (int i = 0; i < num_edges; ++i)
    ++temp_counts[srcs[i]];
  for (int i = 0; i < num_verts; ++i)
    out_degree_list[i+1] = out_degree_list[i] + temp_counts[i];
  copy(out_degree_list, out_degree_list + num_verts, temp_counts);
  for (int i = 0; i < num_edges; ++i)
    out_array[temp_counts[srcs[i]]++] = dsts[i];

  for (int i = 0; i < num_edges; ++i)
    in_array[i] = 0;
  for (int i = 0; i < num_verts+1; ++i)
    in_degree_list[i] = 0;
  for (int i = 0; i < num_verts; ++i)
    temp_counts[i] = 0;

  for (int i = 0; i < num_edges; ++i)
    ++temp_counts[dsts[i]];
  for (int i = 0; i < num_verts; ++i)
    in_degree_list[i+1] = in_degree_list[i] + temp_counts[i];
  copy(in_degree_list, in_degree_list + num_verts, temp_counts);
  for (int i = 0; i < num_edges; ++i)
    in_array[temp_counts[dsts[i]]++] = srcs[i];
  delete [] temp_counts;
}

#define NOT_SET -1
#define IN_SET 0
#define SCC_SET 1
#define OUT_SET 2

void output_connectivity_info(graph* g, int* conn)
{
  int num_in = 0;
  int num_out = 0;
  int num_scc = 0;

#pragma omp parallel for \
  reduction(+:num_in) reduction(+:num_out) reduction(+:num_scc)
  for (int i = 0; i < g->num_verts; ++i) {
    if (conn[i] == IN_SET)
      ++num_in;
    else if (conn[i] == SCC_SET)
      ++num_scc;
    else if (conn[i] == OUT_SET)
      ++num_out;
  }

  printf("Size SCC: %d, In set: %d, Out set: %d\n", num_scc, num_in, num_out);
}


double* do_connectivity_info(graph* g, int root)
{
  int* conn = new int[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    conn[i] = NOT_SET;

  conn[root] = OUT_SET;

    int* seen = new int[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    seen[i] = 0;
  seen[root] = 1;
  
  int* queue = new int[g->num_verts];
  //int *queue = (int *)malloc(sizeof(int)*g->num_verts);
  int* queue_next = new int[g->num_verts];
  //int *queue_next = (int *)malloc(sizeof(int)*g->num_verts);
  int queue_size = 0;
  int next_size = 0;

  double timer = omp_get_wtime();

  queue[queue_size++] = root;
  int updates = 0;
  while (queue_size > 0)
  { 
#pragma omp parallel for
    for (int i = 0; i < queue_size; ++i)
    { 
      int u = queue[i];
      int out_degree = out_degree(g, u);      

      int* out_edges = out_vertices(g, u);
      
      for (int j = 0; j < out_degree; ++j)
      {
        int out = out_edges[j];
        //printf("%d\n", out);
        if (seen[out] == 0)
        {
          conn[out] = OUT_SET; 
          seen[out] = 1;
          int index = 0;
      #pragma omp atomic capture
          index = next_size++;
          queue_next[index] = out;
        }
      }    
    }
    int* tmp = queue;
    queue = queue_next;
    queue_next = tmp;
    queue_size = next_size;
    next_size = 0;
  }
  
  queue_size = 0;

  conn[root] = SCC_SET;
  queue[queue_size++] = root;
  for (int i = 0; i < g->num_verts; ++i)
    seen[i] = 0;
  seen[root] = 1;
  while (queue_size > 0)
  { 
#pragma omp parallel for
    for (int i = 0; i < queue_size; ++i)
    { 
      int u = queue[i];
     
        int in_degree = in_degree(g, u);
        //printf("---%d\n", in_degree);
        int* in_edges = in_vertices(g, u);

        for (int j = 0; j < in_degree; ++j)
        {
          int in = in_edges[j];
          if (seen[in] == 0)
          {
            if ( conn[in] == OUT_SET )
              conn[in] = SCC_SET;
            else if (conn[in] == NOT_SET)
              conn[in] = IN_SET;
            int index = 0;
        #pragma omp atomic capture  
            index = next_size++;
            queue_next[index] = in;
            seen[in] = 1;
          }
        }
    }
    int* ntmp = queue;
    queue = queue_next;
    queue_next = ntmp;
    queue_size = next_size;
    next_size = 0;
  }
  
  
  timer = omp_get_wtime() - timer;

  printf("time: %1.6lf\n", timer);

  output_connectivity_info(g, conn);


  delete [] conn;
  delete [] queue;
  delete [] queue_next;
  return 0;
}


int main(int argc, char** argv)
{
  int* srcs;
  int* dsts;
  int num_verts;
  int num_edges;
  int* out_array;
  int* in_array;
  int* out_degree_list;
  int* in_degree_list;

  if (argc < 2) 
  {
    printf("Usage: %s [graphfile]\n", argv[0]);
    exit(0);
  }

  read_edge(argv[1], num_verts, num_edges, srcs, dsts);
  create_csr(num_verts, num_edges, srcs, dsts, 
    out_array, out_degree_list, in_array, in_degree_list);
  graph g = {num_verts, num_edges, 
             out_array, out_degree_list, in_array, in_degree_list};
  delete [] srcs;
  delete [] dsts;

  int root = 749;
  do_connectivity_info(&g, root);

  delete [] out_array;
  delete [] in_array;
  delete [] out_degree_list;
  delete [] in_degree_list;

  return 0;
}
