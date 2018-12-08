#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

Vec3b Ipoisson, Ialgues;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////


void onMouseIpoisson(int event,int x,int y,int foo,void* p)
{
    if (event!= CV_EVENT_LBUTTONDOWN)
        return;
    Mat* B=(Mat*)p;
    Ipoisson=B->at<Vec3b>(y,x);
    cout<<"Ipoisson ="<<Ipoisson<<endl;
    imshow("Calcul I poisson",*B);
}



void onMouseIalgues(int event,int x,int y,int foo,void* p)
{
    if (event!= CV_EVENT_LBUTTONDOWN)
        return;
    Mat* B=(Mat*)p;
    Ialgues=B->at<Vec3b>(y,x);
    cout<<"Ialgues ="<<Ialgues<<endl;
    imshow("Calcul I algues",*B);
}


void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1); 
	g.add_node(2); 
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}



double fonctionG(double x){
    return 1./(1+x*x);
}

int main() {
	//testGCuts();

	Mat I=imread("../fishes.jpg");
	imshow("I",I);
    //setMouseCallback("Calcul Ipoisson",onMouseIpoisson,&I);
    //setMouseCallback("Calcul Ialgues",onMouseIalgues,&I);
    waitKey();
    int h=I.rows, w=I.cols;
    int n=h*w; //number of nodes
    
    Ipoisson=I.at<Vec3b>(0.3*h, 0.3*w);
    Ialgues=I.at<Vec3b>(0.8*h, 0.4*w);
    
    Graph<double,double,double> g(n,4*n); //8 voisins au maximum pour chacun/divisé par deux
    g.add_node(n);
    int u,v;
    int uNeighbors[4], vNeighbors[4];//coordonnées des potentiels 4 noeuds voisins, on en prend 4 pour ne pas se répéter
    double lambdaPQ;
    
    //calcul gradient
    Mat Ig;
    cvtColor(I,Ig,CV_BGR2GRAY);
    imshow("images",Ig);waitKey();
    Mat Ix(h,w,CV_32F),Iy(h,w,CV_32F),G(h,w,CV_32F);
    for (int i=0;i<h;i++) {
        for (int j=0;j<w;j++){
            float ix,iy;
            if (i==0 || i==h-1)
                iy=0;
            else
                iy=(float(Ig.at<uchar>(i+1,j))-float(Ig.at<uchar>(i-1,j)))/2;
            if (j==0 || j==w-1)
                ix=0;
            else
                ix=(float(Ig.at<uchar>(i,j+1))-float(Ig.at<uchar>(i,j-1)))/2;
            Ix.at<float>(i,j)=ix;
            Iy.at<float>(i,j)=iy;
            G.at<float>(i,j)=sqrt(ix*ix+iy*iy);
        }
    }
    for (int i=0; i<n-1; i++){
        v=i/w;
        u=i%w;
        double Dp0=norm(I.at<Vec3b>(v,u)-Ipoisson);
        double Dp1=norm(I.at<Vec3b>(v,u)-Ialgues);
        g.add_tweights( i, Dp1, Dp0 );
        //trouver 4 voisins en faisant tests si voisin valide : INDICES, et calculer lambdas
        uNeighbors[0]=u+1,vNeighbors[0]=v-1;
        uNeighbors[1]=u+1,vNeighbors[1]=v;
        uNeighbors[2]=u+1,vNeighbors[2]=v+1;
        uNeighbors[3]=u,vNeighbors[3]=v+1;
        for (int r=0; r<4; r++){
            if (uNeighbors[r]>=0 && uNeighbors[r]<w && vNeighbors[r]>=0 && vNeighbors[r]<h){
                int uBis=uNeighbors[r], vBis=vNeighbors[r];
                lambdaPQ=(fonctionG(G.at<float>(v,u))+fonctionG(G.at<float>(vBis,uBis)))/2.;
                int iNeighborR=i+(vBis-v)*w+(uBis-u);
                g.add_edge(i,iNeighborR,lambdaPQ,lambdaPQ);
            }
        }
    }
    double flow=g.maxflow();
    Mat A(h,w,CV_32F);
    //refaire boucle pour trouver g.what_segment
    for (int i=0; i<n-1; i++){
        v=i/w;
        u=i%w;
        if (g.what_segment(i) == Graph<double,double,double>::SOURCE){
            A.at<float>(v,u)=1;
        }
        else{
            A.at<float>(v,u)=0;
        } 
    }
    imshow("A",A);
    waitKey(0);
	return 0;
}


