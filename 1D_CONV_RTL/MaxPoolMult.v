module MaxPoolMult(clk, reset, mpInput, mpOutput, done_flag);
parameter DATA_WIDTH = 16;
parameter K = 4;
parameter W = 128;

input reset,clk;
input [0:W*K*DATA_WIDTH-1] mpInput;
output reg [0:(W/2)*K*DATA_WIDTH-1] mpOutput;
output reg done_flag;

reg [0:W*DATA_WIDTH-1] mpInput_s;
wire [0:(W/2)*DATA_WIDTH-1] mpOutput_s;
reg [3:0] counter;

maxPoolSingle
  #(
      .DATA_WIDTH(DATA_WIDTH),
      .InputW(W)
  ) maxPool
  (
      .mPoolIn(mpInput_s),
      .mPoolOut(mpOutput_s)
  );

always @ (posedge clk) begin
  if (reset == 1'b0) begin
    counter <= 0;
    done_flag <= 1'b0;
    mpInput_s <= 0;
    mpOutput <= 0;
  end else if(counter<5) begin 
    mpInput_s <= mpInput[counter*W*DATA_WIDTH+:W*DATA_WIDTH];
    mpOutput[(counter-1)*(W/2)*DATA_WIDTH+:(W/2)*DATA_WIDTH] <= mpOutput_s;
    counter <= counter+1;
  end else if(counter == 5) begin
    done_flag <= 1'b1;
    mpInput_s <= mpInput_s;
  end
end

//always@(posedge clk)begin
//    if(reset == 0)begin
//        //mpInput_s <= 0;
//        mpOutput <= 0;
//    end else begin
//        //mpInput_s <= mpInput[counter*W*DATA_WIDTH+:W*DATA_WIDTH];
//        mpOutput[(counter-1)*(W/2)*DATA_WIDTH+:(W/2)*DATA_WIDTH] <= mpOutput_s;    
//    end
//end    

endmodule

