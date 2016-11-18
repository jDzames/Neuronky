package networks;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.temporal.TemporalDataDescription;
import org.encog.ml.data.temporal.TemporalMLDataSet;
import org.encog.ml.data.temporal.TemporalPoint;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;


@SuppressWarnings("Duplicates")
public class RecurrentCtypeSiet {

	public String fileName = "C:\\Users\\jDzama\\workspaceEE\\Neuronky\\copy_omni2_24739.lst.txt";
    public File file = new File(fileName);
    public int pocetRiadkov = 17558;
    public int startOf2009 = 8796;

	public double input[][] ;
	public double outputIdeal[][];

    public final int INPUT_WINDOW_SIZE = 8;
    public final int PREDICT_WINDOW_SIZE = 1;

	public double max1_scaB = 21.7;
	public double min1_scaB = 0.4;
    public NormalizedField norm1_scaB = new NormalizedField(
            NormalizationAction.Normalize, "scaB", max1_scaB, min1_scaB, 1, 0);

	public double max2_bzGSE = 23.0;
	public double min2_bzGSE = 0.0;
    public NormalizedField norm2_bzGSE = new NormalizedField(
            NormalizationAction.Normalize, "bzGSE", max2_bzGSE, min2_bzGSE, 1, 0);

	public double max3_bzGSM = 13.5;
	public double min3_bzGSM = -14.3;
    public NormalizedField norm3_bzGSM = new NormalizedField(
            NormalizationAction.Normalize, "bzGSM", max3_bzGSM, min3_bzGSM, 1, 0);

	public double max4_rms = 21.7;
	public double min4_rms = 0.4;
    public NormalizedField norm4_rms = new NormalizedField(
            NormalizationAction.Normalize, "rms", max4_rms, min4_rms, 1, 0);

	public double max5_proton = 999.9;
	public double min5_proton = 0.6;
    public NormalizedField norm5_proton = new NormalizedField(
            NormalizationAction.Normalize, "proton", max5_proton, min5_proton, 1, 0);

	public double max_dst = 43.0;
	public double min_dst = -79.0;
    public NormalizedField norm_dst = new NormalizedField(
            NormalizationAction.Normalize, "dst", max_dst, min_dst, 1, 0);


    public static double normalize(double max, double min, double x){
        return 2*( (x - min)/(max-min) ) -1;
    }


    public MLDataSet createTraining(File rawFile) {


        TemporalMLDataSet trainingData = initDataSet();
        MLDataSet trainSet = trainingData;
        try(BufferedReader br = new BufferedReader(new FileReader(file))){

            String line1 = br.readLine();
            String[] line1Arr = line1.split(" ");
            String line2 = br.readLine();
            String[] line2Arr = line2.split(" ");
            String line3 = br.readLine();
            String[] line3Arr = line3.split(" ");

            //vytvorim si 3 body, ktore postupne budem posuvat, kvoli tomu
            //ze vysledok je az dalej v datach (o 3 nizsie)
            int sequenceNumber1 = Integer.parseInt(line1Arr[0])*100000 + Integer.parseInt(line1Arr[1])*100 + Integer.parseInt(line1Arr[2]);
            TemporalPoint point1 = new TemporalPoint(trainingData
                    .getDescriptions().size());
            point1.setSequence(sequenceNumber1);
            point1.setData(0, norm1_scaB.normalize(Double.parseDouble(line1Arr[3])) );
            point1.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line1Arr[4])) );
            point1.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line1Arr[5])) );
            point1.setData(3, norm4_rms.normalize(Double.parseDouble(line1Arr[6])) );
            point1.setData(4, norm5_proton.normalize(Double.parseDouble(line1Arr[7])) );

            int sequenceNumber2 = Integer.parseInt(line2Arr[0])*100000 + Integer.parseInt(line2Arr[1])*100 + Integer.parseInt(line2Arr[2]);
            TemporalPoint point2 = new TemporalPoint(trainingData
                    .getDescriptions().size());
            point2.setSequence(sequenceNumber2);
            point2.setData(0, norm1_scaB.normalize(Double.parseDouble(line2Arr[3])) );
            point2.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line2Arr[4])) );
            point2.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line2Arr[5])) );
            point2.setData(3, norm4_rms.normalize(Double.parseDouble(line2Arr[6])) );
            point2.setData(4, norm5_proton.normalize(Double.parseDouble(line2Arr[7])) );

            int sequenceNumber3 = Integer.parseInt(line3Arr[0])*100000 + Integer.parseInt(line3Arr[1])*100 + Integer.parseInt(line3Arr[2]);
            TemporalPoint point3 = new TemporalPoint(trainingData
                    .getDescriptions().size());
            point3.setSequence(sequenceNumber3);
            point3.setData(0, norm1_scaB.normalize(Double.parseDouble(line3Arr[3])) );
            point3.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line3Arr[4])) );
            point3.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line3Arr[5])) );
            point3.setData(3, norm4_rms.normalize(Double.parseDouble(line3Arr[6])) );
            point3.setData(4, norm5_proton.normalize(Double.parseDouble(line3Arr[7])) );

            for (int i = 3; i < startOf2009+3; i++) {
                String line4 = br.readLine();
                String[] line4Arr = line4.split(" ");

                //ukoncim prvy point s vysledkom o 3 dalej a vlozim ho do trenovacich dat
                point1.setData(5, norm_dst.normalize(Double.parseDouble(line4Arr[8])) );
                trainingData.getPoints().add(point1);

                //shiftnem data
                point1 = point2;
                point2 = point3;
                //pre lahsie nacitanie (plnim point3), prehladnost
                line3Arr = line4Arr;

                //nacitam data do pointu3 (stary point3 je uz v 2)
                sequenceNumber3 = Integer.parseInt(line3Arr[0])*100000 + Integer.parseInt(line3Arr[1])*100 + Integer.parseInt(line3Arr[2]);
                point3 = new TemporalPoint(trainingData
                        .getDescriptions().size());
                point3.setSequence(sequenceNumber3);
                point3.setData(0, norm1_scaB.normalize(Double.parseDouble(line3Arr[3])) );
                point3.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line3Arr[4])) );
                point3.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line3Arr[5])) );
                point3.setData(3, norm4_rms.normalize(Double.parseDouble(line3Arr[6])) );
                point3.setData(4, norm5_proton.normalize(Double.parseDouble(line3Arr[7])) );

            }


        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            trainingData.generate();

        }
        return trainSet;
    }

    public void train(BasicNetwork network,MLDataSet training, double error)
    {
        final Train train = new ResilientPropagation(network, training);

        int epoch = 1;

        do {
            train.iteration();
            System.out
                    .println("Epoch #" + epoch + " Error:" + train.getError());
            epoch++;
        } while(train.getError() > error);

        System.out.println( network.calculateError(training));
    }



    public TemporalMLDataSet initDataSet() {
        // create a temporal data set
        TemporalMLDataSet dataSet = new TemporalMLDataSet(INPUT_WINDOW_SIZE, PREDICT_WINDOW_SIZE);

        // we are dealing with two types of columns.
        // The first is the both an input (used to
        // predict) and an output (we want to predict it), so true,true.
        TemporalDataDescription dstDescription = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, true);

        // The second is
        // input (used to predict) only, so true,false.
        TemporalDataDescription scaB1Desc = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription bzGSE2Desc = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription bzGSM3Desc = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription rms4Desc = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);
        TemporalDataDescription proton5Desc = new TemporalDataDescription(TemporalDataDescription.Type.RAW, true, false);

        dataSet.addDescription(scaB1Desc);
        dataSet.addDescription(bzGSE2Desc);
        dataSet.addDescription(bzGSM3Desc);
        dataSet.addDescription(rms4Desc);
        dataSet.addDescription(proton5Desc);
        dataSet.addDescription(dstDescription);
        return dataSet;
    }





    public MLDataSet predict(BasicNetwork network) {
        // You can also use the TemporalMLDataSet for prediction.  We will not use "generate"
        // as we do not want to generate an entire training set.  Rather we pass it each part of data
        // and it will produce the input to the model, once there is enough data.
        TemporalMLDataSet predictingData = initDataSet();

        BasicNetwork regular = (BasicNetwork)network.clone();

        regular.clearContext();

        try(BufferedReader br = new BufferedReader(new FileReader(file))){

            //posuniem sa na dalsi rok
            for (int i=0; i< startOf2009-8;i++) br.readLine();

            String line1 = br.readLine();
            String[] line1Arr = line1.split(" ");
            String line2 = br.readLine();
            String[] line2Arr = line1.split(" ");
            String line3 = br.readLine();
            String[] line3Arr = line1.split(" ");

            //vytvorim si 3 body, ktore postupne budem posuvat, kvoli tomu
            //ze vysledok je az dalej v datach (o 3 nizsie)
            int sequenceNumber1 = Integer.parseInt(line1Arr[0])*100000 + Integer.parseInt(line1Arr[1])*100 + Integer.parseInt(line1Arr[2]);
            TemporalPoint point1 = new TemporalPoint(predictingData
                    .getDescriptions().size());
            point1.setSequence(sequenceNumber1);
            point1.setData(0, norm1_scaB.normalize(Double.parseDouble(line1Arr[3])) );
            point1.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line1Arr[4])) );
            point1.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line1Arr[5])) );
            point1.setData(3, norm4_rms.normalize(Double.parseDouble(line1Arr[6])) );
            point1.setData(4, norm5_proton.normalize(Double.parseDouble(line1Arr[7])) );

            int sequenceNumber2 = Integer.parseInt(line2Arr[0])*100000 + Integer.parseInt(line2Arr[1])*100 + Integer.parseInt(line2Arr[2]);
            TemporalPoint point2 = new TemporalPoint(predictingData
                    .getDescriptions().size());
            point2.setSequence(sequenceNumber2);
            point2.setData(0, norm1_scaB.normalize(Double.parseDouble(line2Arr[3])) );
            point2.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line2Arr[4])) );
            point2.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line2Arr[5])) );
            point2.setData(3, norm4_rms.normalize(Double.parseDouble(line2Arr[6])) );
            point2.setData(4, norm5_proton.normalize(Double.parseDouble(line2Arr[7])) );

            int sequenceNumber3 = Integer.parseInt(line3Arr[0])*100000 + Integer.parseInt(line3Arr[1])*100 + Integer.parseInt(line3Arr[2]);
            TemporalPoint point3 = new TemporalPoint(predictingData
                    .getDescriptions().size());
            point3.setSequence(sequenceNumber3);
            point3.setData(0, norm1_scaB.normalize(Double.parseDouble(line3Arr[3])) );
            point3.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line3Arr[4])) );
            point3.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line3Arr[5])) );
            point3.setData(3, norm4_rms.normalize(Double.parseDouble(line3Arr[6])) );
            point3.setData(4, norm5_proton.normalize(Double.parseDouble(line3Arr[7])) );

            double oldDst3=0;
            double oldDst2=0;
            double oldDst1=0;

            //aby som vedel, kedy mam dost na test ci je burka (3 do buducna)
            int counter = 0;

            for (int i = startOf2009-8; i < pocetRiadkov-15; i++) {
                // do we have enough data for a prediction yet?
                if( predictingData.getPoints().size()>=predictingData.getInputWindowSize() ) {
                    counter++;
                    // Make sure to use index 1, because the temporal data set is always one ahead
                    // of the time slice its encoding.  So for RAW data we are really encoding 0.
                    MLData modelInput = predictingData.generateInputNeuralData(1);
                    MLData modelOutput = regular.compute(modelInput);
                    double dst = norm_dst.deNormalize(modelOutput.getData(0));
                    //System.out.println(":Predicted = " + dst + "  ,   Actual = " + line3Arr[8] );

                    //zistim ci je burka
                    if (counter > 3 ){
                        if (Math.abs(dst-oldDst3) > 40.0) {
                            System.out.println("BURKA na datume  :::  " + line3Arr[0] + " " + line3Arr[1] + " " + line3Arr[2]);
                        }
                    }

                    //posun v oldDst aby som mal stare data
                    oldDst3 = oldDst2;
                    oldDst2 = oldDst1;
                    oldDst1 = dst;

                    // Remove the earliest training element.  Unlike when we produced training data,
                    // we do not want to build up a large data set.  We just add enough data points to produce
                    // input to the model.
                    predictingData.getPoints().remove(0);


                }

                //nacitavanie vstupov, ako pri treningu
                String line4 = br.readLine();
                String[] line4Arr = line4.split(" ");

                //ukoncim prvy point s vysledkom o 3 dalej a vlozim ho do trenovacich dat
                point1.setData(5, norm_dst.normalize(Double.parseDouble(line4Arr[8])) );
                predictingData.getPoints().add(point1);

                //shiftnem data
                point1 = point2;
                point2 = point3;
                //pre lahsie nacitanie (plnim point3), prehladnost
                line3Arr = line4Arr;

                //nacitam data do pointu3 (stary point3 je uz v 2)
                sequenceNumber3 = Integer.parseInt(line3Arr[0])*100000 + Integer.parseInt(line3Arr[1])*100 + Integer.parseInt(line3Arr[2]);
                point3 = new TemporalPoint(predictingData
                        .getDescriptions().size());
                point3.setSequence(sequenceNumber3);
                point3.setData(0, norm1_scaB.normalize(Double.parseDouble(line3Arr[3])) );
                point3.setData(1, norm2_bzGSE.normalize(Double.parseDouble(line3Arr[4])) );
                point3.setData(2, norm3_bzGSM.normalize(Double.parseDouble(line3Arr[5])) );
                point3.setData(3, norm4_rms.normalize(Double.parseDouble(line3Arr[6])) );
                point3.setData(4, norm5_proton.normalize(Double.parseDouble(line3Arr[7])) );

            }


        } catch (IOException e) {
            e.printStackTrace();
        }


        // generate the time-boxed data
        predictingData.generate();
        return predictingData;
    }


    public BasicNetwork createNetwork(int hiddenNeurons)
    {
        ElmanPattern pattern = new ElmanPattern();
        pattern.setInputNeurons(48);
        pattern.addHiddenLayer(hiddenNeurons);
        pattern.setOutputNeurons(1);
        pattern.setActivationFunction(new ActivationSigmoid());
        return (BasicNetwork)pattern.generate();
    }
	
	public static void main(final String args[]) {
		
		RecurrentCtypeSiet siet = new RecurrentCtypeSiet();
		//siet.vypisMaxMinBurky();

        try {

            BasicNetwork network = siet.createNetwork(120);  // fajn:   100,
            MLDataSet training = siet.createTraining(siet.file);
            siet.train(network,training,0.007);               //fajn :0.007,
            siet.predict(network);

        } catch (Exception ex) {
            ex.printStackTrace();
        }

	}


}
