import * as tf from '@tensorflow/tfjs';

export class test {
    public name: string = '张三';

    constructor() {
        this.init();
    }

    public init() {
        tf.tensor([1, 2, 3, 4], [2, 2], 'int').print();
    }
}

const t = new test();