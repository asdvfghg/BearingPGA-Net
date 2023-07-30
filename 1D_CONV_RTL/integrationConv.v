module integrationConv (clk,reset,CNNinput,CNNoutput);//Conv1F,Conv2F,Conv3F,iFCinput,iConvOutput

parameter DATA_WIDTH = 16;
parameter DataInW = 1024;//输入长度
parameter DataInH = 1;//输入高度
parameter ConvOut = 128;//单个卷积核输出
parameter MaxPout = 256;//最大池化输出
parameter Kernel = 4;//卷积核个数
parameter Fliter = 64;//卷积核长度
parameter DepthC = 1;//卷积核深度
parameter IntIn = 256;//输入节点个数
parameter FC_out = 10;//第一层全连接层输出节点个数
parameter Pad = 28;//输入补0个数（左右各补Pad个0）
parameter Stride = 8;//卷积移动步长
parameter SHIFT_BIT = 4;//全连接层数据位宽移位位数
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
    .D(DepthC),//输入的深度
    .H(DataInH),//输入的高度
    .W(DataInW),//输入的宽度
    .F(Fliter),//卷积核长度
    .K(Kernel),//卷积核数量
    .P(Pad),//padding的数量
    .S(Stride))//卷积核移动的步长
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
    if (reset == 1'b0) begin//复位初始化各模块
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
//这里用握手会不会好一点
//always @(posedge clk or negedge reset) begin
//  if (reset == 1'b0) begin//复位初始化各模块
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
//  else if(counter > 274+5+8+6+1 && counter < 274+5+8+6 + IntIn + 10+1)begin//启动FC1，进行第一层全连接层计算
//       FCreset = 1'b1;
//    end
//  else if(counter > 274+5+8+6 + IntIn + 10+1)begin//启动FC1，进行第一层全连接层计算
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