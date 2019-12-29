import React, { Component } from 'react';
import LinerRegression from '../liner-regerssion/LinerRegerssion';
import LogisticRegression from '../logistic-regression/logistic-regression';
import Xor from '../xor/xor';
import Iris from '../iris/iris';

export default class App extends Component {

    /**
     * 绘制内容
     *
     * @returns
     * @memberof App
     */
    public render() {
        const type: string = 'iris';
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
            case 'iris':
                content = <Iris/>
        }
        return (<div className="App">{content}</div>);
    }

}
