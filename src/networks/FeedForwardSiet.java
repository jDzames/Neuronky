package networks;

import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.temporal.TemporalDataDescription;
import org.encog.ml.data.temporal.TemporalMLDataSet;
import org.encog.ml.data.temporal.TemporalPoint;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.factory.MLTrainFactory;
import org.encog.ml.train.MLTrain;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;
import org.encog.util.simple.EncogUtility;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;


//inspiracia/vzor:    https://github.com/encog/encog-java-examples/blob/master/src/main/java/org/encog/examples/neural/predict/sunspot/MultiSunspot.java
@SuppressWarnings("Duplicates")
public class FeedForwardSiet {

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

    /**
     * Create and train a model.  Use Encog factory codes to specify the model type that you want.
     * @param trainingData The training data to use.
     * @param methodName The name of the machine learning method (or model).
     * @param methodArchitecture The type of architecture to use with that model.
     * @param trainerName The type of training.
     * @param trainerArgs Training arguments.
     * @return The trained model.
     */
    public static MLRegression trainModel(
            MLDataSet trainingData,
            String methodName,
            String methodArchitecture,
            String trainerName,
            String trainerArgs,
            double error) {

        // first, create the machine learning method (the model)
        MLMethodFactory methodFactory = new MLMethodFactory();
        MLMethod method = methodFactory.create(methodName, methodArchitecture, trainingData.getInputSize(), trainingData.getIdealSize());

        // second, create the trainer
        MLTrainFactory trainFactory = new MLTrainFactory();
        MLTrain train = trainFactory.create(method,trainingData,trainerName,trainerArgs);
        // reset if improve is less than 1% over 5 cycles
        //nie je to nas pripad asi
        /*if( method instanceof MLResettable && !(train instanceof Backpropagation) ) {
            train.addStrategy(new RequiredImprovementStrategy(500));
        }*/

        // third train the model
        EncogUtility.trainToError(train, error);

        return (MLRegression)train.getMethod();
    }

    public TemporalMLDataSet createTraining(File rawFile) {
        TemporalMLDataSet trainingData = initDataSet();

        try(BufferedReader br = new BufferedReader(new FileReader(file))){

            String line1 = br.readLine();
            String[] line1Arr = line1.split(" ");
            String line2 = br.readLine();
            String[] line2Arr = line1.split(" ");
            String line3 = br.readLine();
            String[] line3Arr = line1.split(" ");

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
            return trainingData;
        }

    }

    public TemporalMLDataSet predict(File rawFile, MLRegression model) {
        // You can also use the TemporalMLDataSet for prediction.  We will not use "generate"
        // as we do not want to generate an entire training set.  Rather we pass it each part of data
        // and it will produce the input to the model, once there is enough data.
        TemporalMLDataSet predictingData = initDataSet();

        try(BufferedReader br = new BufferedReader(new FileReader(file))){

            //posuniem sa na dalsi rok
            for (int i=0; i< startOf2009-8;i++) br.readLine();

            String line1 = br.readLine();
            String[] line1Arr = line1.split(" ");
            String line2 = br.readLine();
            String[] line2Arr = line2.split(" ");
            String line3 = br.readLine();
            String[] line3Arr = line3.split(" ");

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

            //na hladanie burky - musim poznat stare hodnoty (a toto staci, netreba cele pole)
            double oldDst3=0;
            double oldDst2=0;
            double oldDst1=0;
            int counter  = 0;

            for (int i = startOf2009-8; i < pocetRiadkov-15; i++) {

                // do we have enough data for a prediction yet?
                if( predictingData.getPoints().size()>=predictingData.getInputWindowSize() ) {
                    //zvysim counter
                    counter++;
                    // Make sure to use index 1, because the temporal data set is always one ahead
                    // of the time slice its encoding.  So for RAW data we are really encoding 0.
                    MLData modelInput = predictingData.generateInputNeuralData(1);
                    MLData modelOutput = model.compute(modelInput);
                    double dst = norm_dst.deNormalize(modelOutput.getData(0));
                    //System.out.println(":Predicted = " + dst + "  ,   Actual = " + line3Arr[8] );

                    //test ci je burka
                    if (counter > 3 ){
                        if (Math.abs(dst-oldDst3) > 40.0) {
                            System.out.println("BURKA na datume  :::  " + line3Arr[0] + " " + line3Arr[1] + " " + line3Arr[2]);
                        }
                    }
                    //System.out.println("DATA na datume  ---  " + line3Arr[0] + " " + line3Arr[1] + " " + line3Arr[2] + "  dst = " + dst + "  oldDst = " + oldDst3 +"  rodiel = "+Math.abs(dst-oldDst3));

                    //posun dst hodnot, aby sme mali aj tu spred 3 hodin
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



	
	public static void main(final String args[]) {

        FeedForwardSiet siet = new FeedForwardSiet();
		//siet.vypisMaxMinBurky();

        try {

            // Step 1. Create training data
            TemporalMLDataSet trainingData = siet.createTraining(siet.file);

            // Step 3. Create and train the model.
            // All sorts of models can be used here, see the XORFactory
            // example for more info.
            double error = 0.0005; //pre RPROP ideal  0.001 rychle, 0.0006 ok,
            MLRegression model = trainModel(
                    trainingData,
                    MLMethodFactory.TYPE_FEEDFORWARD,
                    "?:B->SIGMOID->25:B->SIGMOID->?",
                    MLTrainFactory.TYPE_RPROP,
                    "",
                    error);

            // Now predict
            siet.predict(siet.file,model);

            Encog.getInstance().shutdown();

        } catch (Exception ex) {
            ex.printStackTrace();
        }

	}


    public void vypisMaxMinBurky(){
        //max a min klasika, burka vtedy ak pocas 3 hodin pokles o 40/50 dst
        //to treba poriesit, ako si pamatat 3 hodnoty dozadu cislo (1,2,3 a posun stale o 1?)
        //necitam prve tri cisla, to len datum

        try(BufferedReader br = new BufferedReader(new FileReader(file))){
            String line;
            String[] lineArr;
            double dst3=0;
            double dst2=0;
            double dst1=0;
            double dst=0;
            while ((line = br.readLine()) != null){
                lineArr = line.split(" ");
                double sca1 = Double.parseDouble(lineArr[3]);
                double bzGSE2 = Double.parseDouble(lineArr[2]);
                double bzGSM3 = Double.parseDouble(lineArr[5]);
                double rms4 = Double.parseDouble(lineArr[3]);
                double swProton5 = Double.parseDouble(lineArr[7]);

                //test ci je burka a vypis ak ano
                if (Math.abs( dst3-dst) > 40) System.out.println("BURKA na datume  :::  "+lineArr[0]+" "+lineArr[1]+" "+lineArr[2]);

                //posun dst
                dst3 = dst2;
                dst2 = dst1;
                dst1 = dst;
                dst = Double.parseDouble(lineArr[8]);

                //maima a minima
                min1_scaB = sca1 < min1_scaB ? sca1 : min1_scaB;
                max1_scaB = sca1 > max1_scaB ? sca1 : max1_scaB;
                min2_bzGSE = bzGSE2 < min2_bzGSE ? bzGSE2 : min2_bzGSE;
                max2_bzGSE = bzGSE2 > max2_bzGSE ? bzGSE2 : max2_bzGSE;
                min3_bzGSM = bzGSM3 < min3_bzGSM ? bzGSM3 : min3_bzGSM;
                max3_bzGSM = bzGSM3 > max3_bzGSM ? bzGSM3 : max3_bzGSM;
                min4_rms = rms4 < min4_rms ? rms4 : min4_rms;
                max4_rms = rms4 > max4_rms ? rms4 : max4_rms;
                min5_proton = swProton5 < min5_proton ? swProton5 : min5_proton;
                max5_proton = swProton5 > max5_proton ? swProton5 : max5_proton;
                min_dst = dst < min_dst ? dst : min_dst;
                max_dst = dst > max_dst ? dst : max_dst;

            }

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            System.out.println("1 scaB min = "+min1_scaB+"     scaB max = "+max1_scaB);
            System.out.println("2 bzGSE min= "+min2_bzGSE+"     bzGSE max = "+max2_bzGSE);
            System.out.println("3 bzGSM min= "+min3_bzGSM+"     bzGSM max = "+max3_bzGSM);
            System.out.println("4 rms min = "+min4_rms+"     rms max = "+max4_rms);
            System.out.println("5 proton min = "+min5_proton+"     proton max = "+max5_proton);
            System.out.println("dst min = "+min_dst+"     dst max = "+max_dst);

        }
    }


}

/*
* REALNE DATA
*
BURKA na datume  :::  2008 68 15
BURKA na datume  :::  2008 69 6
BURKA na datume  :::  2008 69 7
BURKA na datume  :::  2008 86 13
BURKA na datume  :::  2008 86 14
BURKA na datume  :::  2008 248 4
BURKA na datume  :::  2008 248 5
BURKA na datume  :::  2008 285 10
BURKA na datume  :::  2008 285 11
BURKA na datume  :::  2008 285 12

BURKA na datume  :::  2009 45 9
BURKA na datume  :::  2009 80 13
BURKA na datume  :::  2009 175 22
BURKA na datume  :::  2009 203 5
BURKA na datume  :::  2009 203 6
BURKA na datume  :::  2009 203 7
BURKA na datume  :::  2009 218 7

1 scaB min = 0.4     scaB max = 21.7
2 bzGSE min= 0.0     bzGSE max = 23.0
3 bzGSM max= -14.3     bzGSM max = 13.5
4 rms min = 0.4     rms max = 21.7
5 proton min = 0.6     proton max = 999.9
dst min = -79.0     dst max = 43.0
*
* */

/*
* PREDICTED DATA  0.001
*
* BURKA na datume  :::  2009 203 5
* BURKA na datume  :::  2009 203 6
* */

/*
* PREDICTED DATA 0.0005
*
* BURKA na datume  :::  2009 80 12
* BURKA na datume  :::  2009 175 21
* BURKA na datume  :::  2009 203 4
* BURKA na datume  :::  2009 203 5
* BURKA na datume  :::  2009 218 5
* BURKA na datume  :::  2009 218 6
* */

/*
* PREDICTED DATA 0.00045
*
BURKA na datume  :::  2009 80 12
BURKA na datume  :::  2009 175 20
BURKA na datume  :::  2009 175 21
BURKA na datume  :::  2009 203 4
BURKA na datume  :::  2009 203 5
BURKA na datume  :::  2009 218 5
BURKA na datume  :::  2009 218 6
* */