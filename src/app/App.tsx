import React, { Component } from 'react';
import LinerRegression from '../liner-regerssion/LinerRegerssion';
import LogisticRegression from '../logistic-regression/logistic-regression';
import Xor from '../xor/xor';
import Iris from '../iris/iris';
import Mnist from '../mnist/mnist';

/**
 * 
 *
 * @export
 * @class App
 * @extends {Component}
 */
export default class App extends Component {

    /**
     * 绘制内容
     *
     * @returns
     * @memberof App
     */
    public render() {
        const type: string = 'mnist';
        let content: any;
        switch (type) {
            case 'LinerRegression':// 线性回归
                content = <LinerRegression />;
                break;
            case 'LogisticRegression':// 逻辑回归
                content = <LogisticRegression />;
                break;
            case 'Xor':// Xor
                content = <Xor />;
                break;
            case 'iris':// 二分类数据集，多分类学习
                content = <Iris />;
                break;
            case 'mnist':// 卷积神经网络学习
                content = <Mnist />;
                break;
        }
        return (<div className="App">{content}</div>);
    }

}
