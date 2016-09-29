/*********************************************************
//* Class:  CS 6971 - Parallel Graph Analysis, Fall 2016
//* Instructor: Prof. George M. Slota 
//*
//* Description: Assignment 2: Web Graph Analysis
//*              Part 2: Implementing a Random Surfer 
//* Name:   Wenyin San
//* RIN#:   661476220
//* Email:  sanw@rpi.edu
//*
//* Template: http://www.cs.rpi.edu/~slotag/classes/FA16/assignments/hw02-sruf.cpp            
//* Due Date: Thur 29 Sept. 2016, 16:00
//*********************************************************
/*
Output:
time: 0.013701
Sum: 1.000113 -- Max: 0.007490 -- Vert: 2264
time: 0.003038
Sum: 1.000000 -- Max: 0.006987 -- Vert: 2264
Total error:  0.274806, Avg error:  0.000028
Sum: 1.000000 -- Max: 0.007107 -- Vert: 8226
time: 0.001435
Total error:  0.385698, Avg error:  0.000039
Short Answer Questions
1. 
-- To get a total_error < 0.01 for random surfer, it needs about 10,000,000 iterations, and the number of 
   clicks is about 92,196,352. It is much higher than 50 iterations needed by the standard algorithm.
--  Output for google.graph
    CC info -- Num CCs: 2746, Num Trivial: 0, Max CC: 855802
--  Output for random.graph
    CC info -- Num CCs: 80010, Num Trivial: 10, Max CC: 10
--  Running Time Table:(Algos run separately)
    Time:              google.graph           random.graph
    BFS Serial:        0.114778               0.049212
    BFS OpenMP:        0.117435               0.059967
    BFS MPI:           8.502110               243.983274
    Color Serial:      0.391075               0.168018
    Color OpenMP:      0.116923               0.039672
    Color MPI:         0.524890               0.172590
--  Serial BFS runs fatest for BFS, Color propagation using OpenMP runs fastest for color propagation.    
2. 
--  Yes. 
    Add printf("%d\n", ccLabel[643]); at the end of one algorithm. And the output is 0.
    Therefore, vertice 643 is labeled with 0, which means that 0 and 643 are in the same component.
3. 
--  To turn OpenMP color propagation implementation into a pushing algorithm, push the minimal vertex id to 
    neighbors instead of find minimum of label among the component.
    Initialize a queue Q, and add all verts to Q. 
    For all v in Q, visit v, remove v from Q. Push the label of v to all of its neighbors.
--  Race conditions might arise from both v and u in Q pushing the label to one vertex.
*/

#include <mpi.h>
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

  double timer = omp_get_wtime();

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

  if (rank == 0) 
  {
    timer = omp_get_wtime() - timer;
    printf("time: %1.6lf\n", timer);
    do_pagerank_verification(g, pageranks);
  }
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
  l = OUTLINK_PROB*100;
  double timer = omp_get_wtime();

  currPage = rand() % g->num_verts;
  int ct = 0;

  for (int iter = 0; iter < num_iter; ++iter)
  { 
    currPage = rand() % g->num_verts;
    int out_degree = out_degree(g, currPage);
    ++ct;
    ++visit_counts[currPage];
    while ( out_degree > 0 )
    { 
      r = rand()%100;
      if (r < l) 
      { 
        int* out_vertices = out_vertices(g, currPage);
        int randConnection = rand() % out_degree;
        int nextPage = out_vertices[randConnection];
        currPage = nextPage;
      }
      else {
      //go to random page
      int nextPage = rand() % g->num_verts;
      currPage = nextPage;
      }
      out_degree = out_degree(g, currPage);
      ++ct;
      ++visit_counts[currPage];
    }
   
  }
  for (int i = 0; i < g->num_verts; ++i)
  {
    pageranks[i] = 1.0*visit_counts[i]/ct;
  }
  timer = omp_get_wtime() - timer;

  if (rank == 0) 
  {
    printf("time: %1.6lf\n", timer);
    do_pagerank_verification(g, pageranks);
  }
  delete [] visit_counts;

  return pageranks;
}


double* get_pageranks_walk_mpi(graph* g, int num_iter)
{ 

  int start_iter = rank * (num_iter/ size + 1);
  int end_iter = (rank + 1) * (num_iter / size + 1);
  if (end_iter > num_iter)
    end_iter = num_iter;

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
  l = OUTLINK_PROB*100;
  double timer = omp_get_wtime();

  currPage = rand() % g->num_verts;
  int ct = 0;
  for (int iter = start_iter; iter < end_iter; ++iter)
  { 
    currPage = rand() % g->num_verts;
    int out_degree = out_degree(g, currPage);
    ++ct;
    ++visit_counts[currPage];
    while ( out_degree > 0 )
    { 
      r = rand()%100;
      if (r < l) 
      { 
        int* out_vertices = out_vertices(g, currPage);
        int randConnection = rand() % out_degree;
        int nextPage = out_vertices[randConnection];
        currPage = nextPage;
      }
      else {
      //go to random page
      int nextPage = rand() % g->num_verts;
      currPage = nextPage;
      }
      out_degree = out_degree(g, currPage);
      ++ct;
      ++visit_counts[currPage];
    }
   
  }

  MPI_Allreduce(MPI_IN_PLACE, visit_counts, g->num_verts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);


  for (int i = 0; i < g->num_verts; ++i)
  {
    pageranks[i] = 1.0*visit_counts[i]/ct;
  }
  timer = omp_get_wtime() - timer;
  if (rank == 0) 
  {
    do_pagerank_verification(g, pageranks);
    printf("time: %1.6lf\n", timer);
    printf("counter: %d\n", ct);

  }
  delete [] visit_counts;

  return pageranks;
}


int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  srand(time(0) + rank);

  if (argc < 3) 
  {
    if (rank == 0) 
    printf("Usage: %s [graphfile] [num clicks]\n", argv[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
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
  if (rank == 0) 
    calc_error(g.num_verts, pageranks1, pageranks2);

  double* pageranks3 = get_pageranks_walk_mpi(&g, atoi(argv[2]));
  if (rank == 0) 
    calc_error(g.num_verts, pageranks1, pageranks3);

  delete [] pageranks1;
  delete [] pageranks2;
  delete [] pageranks3;

  delete [] out_array;
  delete [] in_array;
  delete [] out_degree_list;
  delete [] in_degree_list;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}