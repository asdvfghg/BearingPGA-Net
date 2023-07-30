module integrationConv (clk,reset,CNNinput,CNNoutput);//Conv1F,Conv2F,Conv3F,iFCinput,iConvOutput

parameter DATA_WIDTH = 16;
parameter DataInW = 1024;//���볤��
parameter DataInH = 1;//����߶�
parameter ConvOut = 128;//������������
parameter MaxPout = 256;//���ػ����
parameter Kernel = 4;//����˸���
parameter Fliter = 64;//����˳���
parameter DepthC = 1;//��������
parameter IntIn = 256;//����ڵ����
parameter FC_out = 10;//��һ��ȫ���Ӳ�����ڵ����
parameter Pad = 28;//���벹0���������Ҹ���Pad��0��
parameter Stride = 8;//����ƶ�����
parameter SHIFT_BIT = 4;//ȫ���Ӳ�����λ����λλ��
input clk, reset;
input [DataInW*DataInH*DATA_WIDTH-1:0] CNNinput;//1024*16=16384
output [3:0] CNNoutput;

//wire [Kernel*Fliter*DepthC*DATA_WIDTH-1:0] ConvF;//4*64*1*16=2400
//wire [Kernel*DATA_WIDTH-1:0] ConvB = 64'b1000000000001111_0000000000000011_1000000000000111_1000000000001011;//6
//wire [FC_out*DATA_WIDTH-1:0] FCBOut;
//wire [DATA_WIDTH*FC_out-1:0] FC_W;
//wire [DATA_WIDTH*FC_out-1:0] FC_B = 160'b1000000000000010_1000000000000100_1000000000000011_1000000000000011_1000000000000001_1000000000000100_1000000000000100_1000000000000011_1000000000000011_1000000000000010;
//reg [8:0] address1;

wire MPstart,FCstart,SFstart,CFstart;//,RELUstart
reg Crst,MPrst,SFreset,FCreset,CFreset;//,Relureset


wire [ConvOut*Kernel*DepthC*DATA_WIDTH-1:0] Cout;//128*4*16
wire [MaxPout*DepthC*DATA_WIDTH-1:0] MPout;//256*16
//wire [MaxPout*DepthC*DATA_WIDTH-1:0] ReluOut;//256*16
wire [MaxPout*DepthC*DATA_WIDTH-1:0] SFOut;//256*16
wire [FC_out*DATA_WIDTH-1:0] FCOut;

convLayerMulti 
#(
    .DATA_WIDTH(DATA_WIDTH),
    .D(DepthC),//��������
    .H(DataInH),//����ĸ߶�
    .W(DataInW),//����Ŀ��
    .F(Fliter),//����˳���
    .K(Kernel),//���������
    .P(Pad),//padding������
    .S(Stride))//������ƶ��Ĳ���
C(
	.clk(clk),
	.reset(Crst),
	.signal(CNNinput),//1024*16
	.outputConv(Cout),
	.done_flag(MPstart));//128*4*16

MaxPoolMult
#(
    .DATA_WIDTH(DATA_WIDTH),
    .K(Kernel),
    .W(ConvOut))
MPANDRELU(
    .clk(clk), 
    .reset(MPrst), 
    .mpInput(Cout), 
    .mpOutput(MPout),
    .done_flag(SFstart));
    
//activationFunction
//#(
//    .DATA_WIDTH(DATA_WIDTH),
//    .OUTPUT_NODES(MaxPout))
//Relu(
//    .clk(clk),
//    .reset(Relureset),
//    .input_fc(MPout),
//    .output_fc(ReluOut),
//    .done_flag(SFstart));


shiftFunction
#(
    .DATA_WIDTH(DATA_WIDTH),
    .OUTPUT_NODES(MaxPout),
    .SHIFT_BIT(SHIFT_BIT)
)SF
(
    .clk(clk),
    .reset(SFreset),
    .input_relu(MPout),
    .output_fc(SFOut), 
    .done_flag(FCstart));

layer
#(.DATA_WIDTH(16),
  .INPUT_NODES(IntIn),
  .OUTPUT_NODES(FC_out),
  .RIGHT_SHIFT_BIT(8))
 FC(
    .clk(clk),
    .reset(FCreset),
    .input_fc(SFOut),
    //.weights(FC_W),
    .output_fc(FCOut),
    .done_flag(CFstart)
    );

//FcBiasAdd 
//#(
//.DATA_WIDTH(DATA_WIDTH),
//.OUTPUT_NODES(FC_out))
//FCBADD(
//    .fc_in(FCOut),
//    .biases(FC_B),
//    .cnn_out(FCBOut));

classification CF(
    .clk(clk),
    .reset(CFreset),
    .datain(FCOut),
    .dataout(CNNoutput));
 
always @(posedge clk) begin// or negedge reset
    if (reset == 1'b0) begin//��λ��ʼ����ģ��
      Crst <= 1'b0;
      MPrst <= 1'b0;
      //Relureset <= 1'b0;  
      FCreset <= 1'b0;
      CFreset <= 1'b0;
      SFreset <= 1'b0;
    end else begin
       if(MPstart == 0 && SFstart == 0 && FCstart == 0 && CFstart == 0)
            Crst <= 1'b1;

       if(MPstart == 1'b1 && SFstart == 0 && FCstart == 0 && CFstart == 0)
            MPrst <= 1'b1;
 
//       if(MPstart == 1'b1 && RELUstart == 1'b1 && SFstart == 0 && FCstart == 0 && CFstart == 0)
//            Relureset <= 1'b1;

       if(MPstart == 1'b1 && SFstart == 1'b1 && FCstart == 0 && CFstart == 0)
            SFreset <= 1'b1;
            
       if(MPstart == 1'b1 && SFstart == 1'b1 && FCstart == 1'b1 && CFstart == 0)
            FCreset <= 1'b1;

       if(MPstart == 1'b1 && SFstart == 1'b1 && FCstart == 1'b1 && CFstart == 1'b1)
            CFreset <= 1'b1;
       else 
            CFreset <= 1'b0;
    end
end

//////////////////////////////////////////////////////////
//���������ֻ᲻���һ��
//always @(posedge clk or negedge reset) begin
//  if (reset == 1'b0) begin//��λ��ʼ����ģ��
//    CM_En=1'b0;
//    Crst = 1'b0;
//    MPrst = 1'b0;
//    Relureset = 1'b0;  
//    FCreset = 1'b0;
//    FCBreset = 1'b0;
//    CFreset = 1'b0;
//    address1 = -1;
//    counter = 0;
//  end
//else begin
//  counter = counter + 1;
//  if(counter > 0 && counter < 5) begin
//       CM_En = 1'b1;
//    end
//  else if (counter > 5 && counter < 274+5) begin
//       Crst = 1'b1;
//    end
//  else if (counter > 274+5 && counter < 274+5+8) begin
//       MPrst = 1'b1;
//    end
//  else if (counter > 274+5+8 && counter <274+5+8+6) begin
//      Relureset = 1'b1;
//    end
//  else if(counter > 274+5+8+6+1 && counter < 274+5+8+6 + IntIn + 10+1)begin//����FC1�����е�һ��ȫ���Ӳ����
//       FCreset = 1'b1;
//    end
//  else if(counter > 274+5+8+6 + IntIn + 10+1)begin//����FC1�����е�һ��ȫ���Ӳ����
//       CFreset = 1'b1;
//    end

//    if(counter >  274+5+8+6)begin
//        if (address1 != 9'h1fe) begin
//           address1 = address1 + 1;
//        end else
//           address1 = 16'h1fe; 
//     end
//     else
//           address1 = -1;
//    end         
//end
endmodule