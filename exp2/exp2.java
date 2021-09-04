package Json;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Vector;

public class exp2{
    public static void main(String[] args) throws Exception{


        File s = new File("D:\\za\\Codeflaws\\low\\output2\\3bench\\bugcode_Coconut_similarities_doc2vec.txt");
        FileReader fr = new FileReader(s);
        BufferedReader br = new BufferedReader(fr);
        String line = "";
        String buff = "";
        line = br.readLine();
        String[] split = line.split(", ");
        Vector<Double> coco = new Vector<Double>();
        for (int i = 0; i < split.length; i++) {
            coco.add(Double.parseDouble(split[i]));
        }

        File s2 = new File("D:\\za\\Codeflaws\\low\\output2\\3bench\\bugcode_GT_similarities_doc2vec.txt");
        fr = new FileReader(s2);
        br = new BufferedReader(fr);
        buff = br.readLine();
        String[] temp = buff.split(", ");
        Vector<Double> gt = new Vector<Double>();
        for (int i = 0; i < temp.length; i++) {
            gt.add(Double.parseDouble(temp[i]));
        }

//        System.out.println(coco.size());
//        System.out.println(gt.size());



        File s3 = new File("D:\\za\\Codeflaws\\low\\num.txt");
        fr = new FileReader(s3);
        br = new BufferedReader(fr);

        Vector<Integer> num = new Vector<Integer>();
        while ((buff = br.readLine()) != null){
            num.add(Integer.parseInt(buff));
        }

        //System.out.println(num.size());

        File s4 = new File("D:\\za\\Codeflaws\\low\\Trex实验数据\\新lable\\lable.txt");
        fr = new FileReader(s4);
        br = new BufferedReader(fr);
        Vector<String> lable = new Vector<>();

        while ((buff = br.readLine()) != null){
            lable.add(buff);
        }




        Vector<Double> threshold = new Vector<>();
        threshold.add(0.909);
        threshold.add(0.364);
        threshold.add(0.891);
        threshold.add(0.917);
        threshold.add(0.939);










        for (int k = 0; k < threshold.size(); k++) {
            Double t = threshold.get(k);
            int PT = 0;
            int NT = 0;
            int PF = 0;
            int NF = 0;
            for (int j = 0; j < num.size(); j++) {





                //int i = num.get(j);
                int i = num.get(j);
                if(coco.get(i) > t ){
                    if(lable.get(j).equals("1")){
                    //System.out.println(i+1 + " postrue" + " " + coco.get(i) +" " + gt.get(i));
                    PT++;
                }

                    else{
                    //System.out.println(i+1 + " negtrue" + " " + coco.get(i) +" " + gt.get(i));
                    PF++;
                }

                }

                else{
                    if(lable.get(j).equals("1")){
                        //System.out.println(i+1 + " negfalse" + " " + coco.get(i) +" " + lable.get(i));
                        NF++;
                    }

                    else{
                       // System.out.println(i+1 + " posfalse" + " " + coco.get(i) +" " + gt.get(i));
                        NT++;
                    }

                }

            }


            double Acc = ((double) PT+(double) NT)/212;
            double Pre = (double)PT/((double)PT+(double)PF);
            double Recall = (double)PT/((double)PT+(double)NF);
            double F1 = 2*Pre*Recall/(Pre+Recall);




            System.out.println(t+"--------------------------------------");
            System.out.println(PT);
            System.out.println(NT);
            System.out.println(PF);
            System.out.println(NF);

            System.out.println( Acc);
            System.out.println(Pre);
            System.out.println(Recall);
            System.out.println( F1);

        }




    }
}
