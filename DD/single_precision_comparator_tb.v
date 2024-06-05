module single_precision_comparator_tb;

    reg [31:0] A_32;
    reg [31:0] B_32;
    wire equal_to;
    wire less_than;
    wire greater_than;

    real A;
    real B;

    single_precision_comparator uut (
        .A_32(A_32),
        .B_32(B_32),
        .equal_to(equal_to),
        .less_than(less_than),
        .greater_than(greater_than)
    );

    // Provide input values and monitor results
    initial begin
        // Test Case 1: Equal numbers (Positives)
        A = 5.4;
        B = 5.4;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $display("Test Case 1:");
        $display("Number1: %h", A_32);
        $display("Number2: %h", B_32);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 2: Equal numbers (Negatives)
        A = -5.4;
        B = -5.4;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $display("Test Case 2:");
        $display("Number1: %h", A_32);
        $display("Number2: %h", B_32);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;


        // Test Case 3: A > B (Positives)
        A = 7.2;
        B = 6.3;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 3:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 4: A > B (Negatives)
        A = -6.3;
        B = -7.2;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 4:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 5: A < B (Positives)
        A = 8.1;
        B = 9.0;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 5:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 6: A < B (Negatives)
        A = -9.0;
        B = -8.1;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 6:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 7: Zeros
        A = 0.0;
        B = 0.0;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 7:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 8: Infinity
        A = 0/0;
        B = 0/0;
        A_32 = $realtobits(A);
        B_32 = $realtobits(B);
        $monitor("Test Case 8:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 9: NaN (Not a Number)
        A = 32'h7FF8000000000000;
        B = 1;
        A_32 = A;
        B_32 = $realtobits(B);
        $monitor("Test Case 9:");
        $monitor("Number1: %h", A_32);
        $monitor("Number2: %h", B_32);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;
        // Finish simulation
        $finish;
    end
endmodule
