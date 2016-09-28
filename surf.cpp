
//#include <mpi.h>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>

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

void calc_error(int arr_len, double* arr1, double* arr2)
{
  double error_sum = 0.0;

  for (int i = 0; i < arr_len; ++i)
    error_sum += fabs(arr1[i] - arr2[i]);

  printf("Total error: %9.6lf, Avg error: %9.6lf\n", 
    error_sum, error_sum / (double)arr_len);
}



void do_pagerank_verification(graph* g, double* pageranks)
{
  double pr_sum = 0.0;
  double max_pr = 0.0;
  int max_vert = -1;
  for (int i = 0; i < g->num_verts; ++i)
  {
    pr_sum += pageranks[i];
    if (pageranks[i] > max_pr) {
      max_pr = pageranks[i];
      max_vert = i;
    } 
  }

  printf("Sum: %1.6lf -- Max: %1.6lf -- Vert: %d\n", pr_sum, max_pr, max_vert);
}


#define OUTLINK_PROB 0.85

double* get_pageranks_calc(graph* g, int num_iter)
{
  double* pageranks = new double[g->num_verts];
  double* pageranks_next = new double[g->num_verts];
  double sum_sinks = 0.0;
  double sum_sinks_next = 0.0;
  
  for (int vert = 0; vert < g->num_verts; ++vert) {
    pageranks[vert] = 1 / (double)g->num_verts;
    if (out_degree(g, vert) == 0)
      sum_sinks += pageranks[vert];
  }
  // calc pageranks
  for (int iter = 0; iter < num_iter; ++iter) {
#pragma omp parallel for schedule(guided) reduction(+:sum_sinks_next)
    for (int vert = 0; vert < g->num_verts; ++vert) {
      double new_pagerank = sum_sinks / (double)g->num_verts;

      int in_degree = in_degree(g, vert);
      int* in_vertices = in_vertices(g, vert);
      for (int j = 0; j < in_degree; ++j)
        new_pagerank += 
          pageranks[in_vertices[j]] / (double)out_degree(g, in_vertices[j]);

      new_pagerank *= OUTLINK_PROB;
      new_pagerank += ((1.0 - OUTLINK_PROB) / (double)g->num_verts);
      if (out_degree(g, vert) == 0)
        sum_sinks_next += pageranks[vert];

      pageranks_next[vert] = new_pagerank;
    }

    double* temp = pageranks;
    pageranks = pageranks_next;
    pageranks_next = temp;
    sum_sinks = sum_sinks_next;
    sum_sinks_next = 0.0;
  }

  //if (rank == 0) 
  do_pagerank_verification(g, pageranks);
  delete [] pageranks_next;

  return pageranks;
}

double* get_pageranks_walk(graph* g, int num_iter)
{
  double* pageranks = new double[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    pageranks[i] = 0.0;
  int* visit_counts = new int[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    visit_counts[i] = 0;
  srand((int) time (0)); 
  //double timer = omp_get_wtime();
  int currPage = 0; 
  int r = 0;
  int l = 0;
  l = (int)OUTLINK_PROB*100;
  currPage = rand() % g->num_verts;
  for (int iter = 0; iter < num_iter; ++iter)
  { 
    ++visit_counts[currPage];
    //for (int vert = 0; vert < g->num_verts; ++vert) 
    //{
    //currPage = vert;
    r = rand()%100;
    if (r < 85) 
    { 
      //go to random connected page
      int out_degree = out_degree(g, currPage);
      //printf("%d\n",out_degree );
      int* out_vertices = out_vertices(g, currPage);
      if(out_degree>0)
      {
        int randConnection = rand() % out_degree;
        int nextPage = out_vertices[randConnection];
      }
      else {
        //go to random page
        int nextPage = rand() % g->num_verts;
        currPage = nextPage;
      }
    }
    else {
      //go to random page
      int nextPage = rand() % g->num_verts;
      currPage = nextPage;
    }
    //}
  }
  for (int i = 0; i < g->num_verts; ++i)
  {
    pageranks[i] = 1.0*visit_counts[i]/(num_iter);
  }

  //if (rank == 0) 
  do_pagerank_verification(g, pageranks);
  delete [] visit_counts;

  return pageranks;
}


double* get_pageranks_walk_mpi(graph* g, int num_iter)
{
  double* pageranks = new double[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    pageranks[i] = 0.0;
  int* visit_counts = new int[g->num_verts];
  for (int i = 0; i < g->num_verts; ++i)
    visit_counts[i] = 0;
  
  
  //if (rank == 0) 
  do_pagerank_verification(g, pageranks);
  delete [] visit_counts;

  return pageranks;
}


int main(int argc, char** argv)
{
  //MPI_Init(&argc, &argv);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &size);
  //srand(time(0) + rank);

  if (argc < 3) 
  {
    //if (rank == 0) 
    printf("Usage: %s [graphfile] [num clicks]\n", argv[0]);
    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int* srcs;
  int* dsts;
  int num_verts;
  int num_edges;
  int* out_array;
  int* in_array;
  int* out_degree_list;
  int* in_degree_list;

  read_edge(argv[1], num_verts, num_edges, srcs, dsts);
  create_csr(num_verts, num_edges, srcs, dsts, 
    out_array, out_degree_list, in_array, in_degree_list);
  graph g = {num_verts, num_edges, 
             out_array, out_degree_list, in_array, in_degree_list};
  delete [] srcs;
  delete [] dsts;

  double* pageranks1 = get_pageranks_calc(&g, 50);
  double* pageranks2 = get_pageranks_walk(&g, atoi(argv[2]));
  //if (rank == 0) 
    calc_error(g.num_verts, pageranks1, pageranks2);

  //double* pageranks3 = get_pageranks_walk_mpi(&g, atoi(argv[2]));
  //if (rank == 0) 
  // calc_error(g.num_verts, pageranks1, pageranks3);

  delete [] pageranks1;
  delete [] pageranks2;
  //delete [] pageranks3;

  delete [] out_array;
  delete [] in_array;
  delete [] out_degree_list;
  delete [] in_degree_list;

  //MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Finalize();

  return 0;
}
